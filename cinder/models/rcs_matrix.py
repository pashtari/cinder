"""Row-contiguous sparse (RCS) matrices.

An RCS matrix is an ``M x N`` sparse matrix in which the non-zero elements of
each row form a contiguous segment of length at most ``L``. It is stored
compactly as a dense ``(M, L)`` matrix of segment values and an ``(M,)``
integer vector of segment start columns, and multiplies dense matrices
directly in this compact form.

On CUDA the product ``A @ B`` is backed by three Triton kernels wrapped in a
``torch.autograd.Function`` (:class:`RCSMatmulFn`):

- forward:  ``out[i, k] = sum_l values[i, l] * B[col_start_i + l, k]``;
  2D grid over (M-blocks, K-blocks), ``l`` loop unrolled over constexpr
  ``L``, gathered rows of ``B``, fp32 accumulator (fp64 for double inputs).
- grad_values: 1D grid over M-blocks; each ``grad_out`` tile is loaded once
  and reused for all ``L`` row-dot accumulators, so ``grad_out`` is streamed
  from DRAM exactly once.
- grad_B: atomic-free segmented reduction over a precomputed transpose
  structure. ``col_start`` is static per matrix, so the stable sort of the
  flattened column targets (:func:`_build_transpose_cache`) is done once and
  reused every backward; each program owns one ``grad_B`` row and K-tile and
  walks its contributor segment in a fixed order, so the result is
  **bit-deterministic** across runs.

First-order backward uses the kernels. When autograd runs backward under
grad mode (``create_graph=True``), the backward switches to a pure-torch
differentiable formulation, so second derivatives are exact and supported.
Under CUDA autocast, fp32 operands are computed in the autocast dtype
(matching what autocast does to ``torch.bmm``) and gradients are cast back.
``L`` is a compile-time constant per kernel specialization; very large ``L``
(hundreds) inflates compile time because the ``l`` loops are fully unrolled.

Without triton (e.g. a CPU-only build), :class:`RCSMatrix` falls back to a
sparse CSR kernel outside autograd and to a differentiable gather-and-contract
path under autograd.
"""

import copy

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only build: RCSMatrix falls back to CSR / gather+bmm
    triton = None

__all__ = ["RCSMatrix"]

_MATMUL_FUNCS = {
    torch.matmul,
    torch.mm,
    Tensor.matmul,
    Tensor.mm,
    Tensor.__matmul__,
}


def _build_transpose_cache(col_start: Tensor, L: int, N: int):
    """One-time transpose structure of the static sparsity pattern.

    Stable sort of the flattened column targets ``J_i + l``; returns
    ``(rowptr (N+1,) int64, src_i (M*L,) int32, src_l (M*L,) int32)`` on
    ``col_start``'s device. Requires ``L > 0``.
    """
    device = col_start.device
    j = col_start.to(torch.long).reshape(-1)
    flat = (j[:, None] + torch.arange(L, device=device, dtype=torch.long)).reshape(-1)
    perm = torch.argsort(flat, stable=True)
    rowptr = torch.searchsorted(
        flat[perm], torch.arange(N + 1, device=device, dtype=torch.long)
    )
    return (
        rowptr.contiguous(),
        (perm // L).to(torch.int32).contiguous(),
        (perm % L).to(torch.int32).contiguous(),
    )


if triton is not None:

    # Tiny autotune space: the tile aspect ratio is the only knob that matters
    # much for this memory-bound gather kernel. Keyed on (K, L) so first-call
    # overhead stays low. Safe to autotune: replaying the forward only rewrites
    # the same output tile (the backward kernels are NOT autotuned).
    _FWD_CONFIGS = [
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 32}, num_warps=4),
    ]

    @triton.autotune(configs=_FWD_CONFIGS, key=["K", "L"])
    @triton.jit
    def _rcs_fwd_kernel(
        g_ptr, j_ptr, b_ptr, out_ptr,
        M, K,
        stride_gm, stride_gl,
        stride_bn, stride_bk,
        stride_om, stride_ok,
        L: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """out[i, k] = sum_l G[i, l] * B[J_i + l, k] over a (BLOCK_M, BLOCK_K) tile."""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_m = rm < M
        mask_k = rk < K
        mask = mask_m[:, None] & mask_k[None, :]
        rm = rm.to(tl.int64)  # 64-bit addressing so large M * strides never overflow
        j = tl.load(j_ptr + rm, mask=mask_m, other=0)

        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=ACC_DTYPE)
        for l in tl.static_range(L):
            g = tl.load(g_ptr + rm * stride_gm + l * stride_gl, mask=mask_m, other=0.0)
            b = tl.load(
                b_ptr + (j + l)[:, None] * stride_bn + rk[None, :] * stride_bk,
                mask=mask,
                other=0.0,
            )
            acc += g.to(ACC_DTYPE)[:, None] * b.to(ACC_DTYPE)

        tl.store(
            out_ptr + rm[:, None] * stride_om + rk[None, :] * stride_ok,
            acc.to(out_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _rcs_bwd_g_kernel(
        go_ptr, j_ptr, b_ptr, gg_ptr,
        M, K,
        stride_gom, stride_gok,
        stride_bn, stride_bk,
        stride_ggm, stride_ggl,
        L: tl.constexpr,
        L_P2: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """grad_G[i, l] = dot(grad_out[i, :], B[J_i + l, :]); grid = (M-blocks,).

        Each grad_out tile is loaded once and reused for all L accumulators
        (a (BLOCK_M, L_P2) register accumulator updated via a constexpr one-hot
        select, which compiles to a predicated register move).
        """
        pid_m = tl.program_id(0)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = rm < M
        rm = rm.to(tl.int64)
        j = tl.load(j_ptr + rm, mask=mask_m, other=0)

        l_idx = tl.arange(0, L_P2)
        acc = tl.zeros((BLOCK_M, L_P2), dtype=ACC_DTYPE)
        for k0 in range(0, K, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            mask_k = rk < K
            mask = mask_m[:, None] & mask_k[None, :]
            go = tl.load(
                go_ptr + rm[:, None] * stride_gom + rk[None, :] * stride_gok,
                mask=mask,
                other=0.0,
            ).to(ACC_DTYPE)
            for li in tl.static_range(L):
                b = tl.load(
                    b_ptr + (j + li)[:, None] * stride_bn + rk[None, :] * stride_bk,
                    mask=mask,
                    other=0.0,
                )
                partial = tl.sum(go * b.to(ACC_DTYPE), axis=1)
                acc = tl.where((l_idx == li)[None, :], acc + partial[:, None], acc)

        tl.store(
            gg_ptr + rm[:, None] * stride_ggm + l_idx[None, :] * stride_ggl,
            acc.to(gg_ptr.dtype.element_ty),
            mask=mask_m[:, None] & (l_idx < L)[None, :],
        )

    @triton.jit
    def _rcs_bwd_b_kernel(
        rowptr_ptr, srci_ptr, srcl_ptr,
        g_ptr, go_ptr, gb_ptr,
        K,
        stride_gm, stride_gl,
        stride_gom, stride_gok,
        stride_gbn, stride_gbk,
        BLOCK_E: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """grad_B[j, k_tile] = sum over segment rowptr[j]..rowptr[j+1] of
        G[src_i, src_l] * grad_out[src_i, k_tile].

        One program per (j, K-tile); fixed traversal order => deterministic.
        """
        j = tl.program_id(0)
        pid_k = tl.program_id(1)
        rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = rk < K
        rk = rk.to(tl.int64)

        start = tl.load(rowptr_ptr + j)
        end = tl.load(rowptr_ptr + j + 1)

        acc = tl.zeros((BLOCK_K,), dtype=ACC_DTYPE)
        for p0 in range(start, end, BLOCK_E):
            re = p0 + tl.arange(0, BLOCK_E)
            mask_e = re < end
            i = tl.load(srci_ptr + re, mask=mask_e, other=0).to(tl.int64)
            l = tl.load(srcl_ptr + re, mask=mask_e, other=0).to(tl.int64)
            g = tl.load(
                g_ptr + i * stride_gm + l * stride_gl, mask=mask_e, other=0.0
            ).to(ACC_DTYPE)
            go = tl.load(
                go_ptr + i[:, None] * stride_gom + rk[None, :] * stride_gok,
                mask=mask_e[:, None] & mask_k[None, :],
                other=0.0,
            ).to(ACC_DTYPE)
            acc += tl.sum(g[:, None] * go, axis=0)

        tl.store(
            gb_ptr + j.to(tl.int64) * stride_gbn + rk * stride_gbk,
            acc.to(gb_ptr.dtype.element_ty),
            mask=mask_k,
        )

    def _acc_tl_dtype(dtype: torch.dtype):
        return tl.float64 if dtype == torch.float64 else tl.float32

    def _launch_forward(values, col_start, B):
        M, L = values.shape
        K = B.shape[1]
        out = torch.empty(M, K, dtype=values.dtype, device=values.device)
        if M == 0 or K == 0:
            return out
        if L == 0:
            return out.zero_()
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(K, meta["BLOCK_K"]))
        _rcs_fwd_kernel[grid](
            values, col_start, B, out,
            M, K,
            values.stride(0), values.stride(1),
            B.stride(0), B.stride(1),
            out.stride(0), out.stride(1),
            L=L, ACC_DTYPE=_acc_tl_dtype(values.dtype),
        )
        return out

    def _launch_grad_g(grad_out, col_start, B, M, L):
        K = grad_out.shape[1]
        grad_g = torch.empty(M, L, dtype=grad_out.dtype, device=grad_out.device)
        if M == 0 or L == 0:
            return grad_g
        if K == 0:
            return grad_g.zero_()
        BLOCK_K = min(128, triton.next_power_of_2(K))
        grid = (triton.cdiv(M, 16),)
        _rcs_bwd_g_kernel[grid](
            grad_out, col_start, B, grad_g,
            M, K,
            grad_out.stride(0), grad_out.stride(1),
            B.stride(0), B.stride(1),
            grad_g.stride(0), grad_g.stride(1),
            L=L, L_P2=triton.next_power_of_2(L),
            BLOCK_M=16, BLOCK_K=BLOCK_K,
            ACC_DTYPE=_acc_tl_dtype(grad_out.dtype),
            num_warps=4,
        )
        return grad_g

    def _launch_grad_b(values, cache, grad_out, N):
        """Deterministic segmented-reduction grad_B.

        Returns grad_B in the fp32/fp64 accumulation dtype (caller casts).
        """
        M, L = values.shape
        K = grad_out.shape[1]
        acc_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
        if M == 0 or K == 0 or L == 0 or N == 0:
            return torch.zeros(N, K, dtype=acc_dtype, device=values.device)
        rowptr, src_i, src_l = cache
        grad_b = torch.empty(N, K, dtype=acc_dtype, device=values.device)
        BLOCK_K = min(128, triton.next_power_of_2(K))
        grid = (N, triton.cdiv(K, BLOCK_K))
        _rcs_bwd_b_kernel[grid](
            rowptr, src_i, src_l,
            values, grad_out, grad_b,
            K,
            values.stride(0), values.stride(1),
            grad_out.stride(0), grad_out.stride(1),
            grad_b.stride(0), grad_b.stride(1),
            BLOCK_E=32, BLOCK_K=BLOCK_K,
            ACC_DTYPE=_acc_tl_dtype(acc_dtype),
            num_warps=4,
        )
        return grad_b

    class RCSMatmulFn(torch.autograd.Function):
        """``out = RCS(values, col_start) @ B`` with Triton forward and backward.

        ``cache`` is the optional transpose structure from
        :func:`_build_transpose_cache` (built on the fly if omitted and grad_B
        is needed; RCSMatrix passes its per-instance cache). Gradients flow to
        ``values`` and ``B`` (only those requested via ``needs_input_grad``
        are computed) and are bit-deterministic. First-order backward uses
        Triton kernels; backward under grad mode (``create_graph=True``) uses
        a differentiable pure-torch formulation, so double backward is
        supported.
        """

        @staticmethod
        @torch.amp.custom_fwd(device_type="cuda")
        def forward(ctx, values: Tensor, col_start: Tensor, B: Tensor, cache=None):
            if not (values.is_cuda and B.is_cuda and col_start.is_cuda):
                raise RuntimeError("triton RCS matmul requires CUDA tensors")

            compute_dtype = torch.promote_types(values.dtype, B.dtype)
            # Emulate autocast's handling of matmul: fp32 operands compute in
            # the autocast dtype (fp64 and explicit low-precision inputs are
            # left alone, as autocast would).
            if compute_dtype == torch.float32 and torch.is_autocast_enabled("cuda"):
                compute_dtype = torch.get_autocast_dtype("cuda")

            col_start = col_start.to(torch.long).contiguous()
            out = _launch_forward(values.to(compute_dtype), col_start, B.to(compute_dtype))

            # Save the ORIGINAL operands: in the common (equal-dtype,
            # no-autocast) case the casts above are no-ops, and the
            # differentiable backward branch re-applies them under grad mode
            # so the graph stays connected.
            ctx.save_for_backward(values, col_start, B)
            ctx.compute_dtype = compute_dtype
            ctx.cache = cache
            return out

        @staticmethod
        @torch.amp.custom_bwd(device_type="cuda")
        def backward(ctx, grad_out: Tensor):
            values, col_start, B = ctx.saved_tensors
            need_values, _, need_b = ctx.needs_input_grad[:3]
            M, L = values.shape
            N = B.shape[0]
            grad_values = grad_b = None

            if torch.is_grad_enabled():
                # create_graph=True: differentiable formulation for exact
                # second derivatives.
                idx = col_start[:, None] + torch.arange(L, device=col_start.device)
                values_c = values.to(grad_out.dtype)
                B_c = B.to(grad_out.dtype)
                if need_values:
                    grad_values = (grad_out.unsqueeze(1) * B_c[idx]).sum(-1).to(values.dtype)
                if need_b:
                    src = (values_c.unsqueeze(-1) * grad_out.unsqueeze(1)).reshape(M * L, -1)
                    grad_b = (
                        torch.zeros_like(B_c).index_add_(0, idx.reshape(-1), src).to(B.dtype)
                    )
                return grad_values, None, grad_b, None

            compute_dtype = ctx.compute_dtype
            go = grad_out.to(compute_dtype)
            if need_values:
                grad_values = _launch_grad_g(go, col_start, B.to(compute_dtype), M, L)
                grad_values = grad_values.to(values.dtype)
            if need_b:
                cache = ctx.cache
                if cache is None and L > 0:
                    cache = _build_transpose_cache(col_start, L, N)
                grad_b = _launch_grad_b(values.to(compute_dtype), cache, go, N)
                grad_b = grad_b.to(B.dtype)
            return grad_values, None, grad_b, None

    def rcs_mm(values: Tensor, col_start: Tensor, B: Tensor, cache=None) -> Tensor:
        """``RCS(values, col_start) @ B`` for dense ``(N, K)`` or vector ``(N,)`` B."""
        if B.ndim == 1:
            return RCSMatmulFn.apply(values, col_start, B.unsqueeze(1), cache).squeeze(1)
        return RCSMatmulFn.apply(values, col_start, B, cache)

else:
    RCSMatmulFn = None
    rcs_mm = None


class RCSMatrix(torch.Tensor):
    """Row-contiguous sparse matrix stored in compact form.

    Represents an ``M x N`` matrix whose row ``i`` is zero except for the
    ``L`` consecutive entries starting at column ``start_cols[i]``, whose
    values are ``values[i]``. Rows with fewer than ``L`` non-zeros are
    zero-padded within their segment.

    The dense matrix is never materialized: a product ``A @ B`` with a dense
    ``(N, K)`` matrix (or ``(N,)`` vector) runs on the compact representation
    in ``O(M L K)`` time and memory instead of ``O(M N K)``. On CUDA the
    product uses the fused Triton kernels defined in this module; without
    triton it falls back to a sparse CSR kernel outside autograd and to a
    differentiable gather-and-contract path under autograd. The product is
    differentiable, with gradients flowing to ``values`` and ``B``, and
    accepts tensor subclasses such as ``nn.Parameter`` on the right-hand
    side. ``pickle`` and ``copy.deepcopy`` round-trip the compact form. Any
    other tensor operation on the matrix raises ``NotImplementedError``;
    convert with :meth:`to_dense` first.

    If ``values`` is swapped to another device in place (as ``module.cuda()``
    does to an ``nn.Parameter``), products and :meth:`to_dense` follow the
    device of ``values``; the wrapper's ``.device`` metadata keeps reporting
    the construction-time device.

    Args:
        values: Non-zero segment values ``G``, of shape ``(M, L)``.
        start_cols: Integer start column ``J`` of each row's segment, of shape
            ``(M,)``, with ``0 <= start_cols[i] <= N - L``. Copied at
            construction; later in-place changes to the argument are ignored.
        num_cols: Number of columns ``N`` of the represented matrix.
    """

    values: Tensor
    start_cols: Tensor

    @staticmethod
    def __new__(cls, values: Tensor, start_cols: Tensor, num_cols: int):
        if values.ndim != 2:
            raise ValueError(f"values must have shape (M, L), got {tuple(values.shape)}")
        if num_cols < 0:
            raise ValueError(f"num_cols must be non-negative, got {num_cols}")
        return torch.Tensor._make_wrapper_subclass(
            cls,
            (values.shape[0], num_cols),
            dtype=values.dtype,
            device=values.device,
            requires_grad=values.requires_grad,
        )

    def __init__(self, values: Tensor, start_cols: Tensor, num_cols: int):
        M, L = values.shape
        if start_cols.shape != (M,):
            raise ValueError(
                f"start_cols must have shape ({M},), got {tuple(start_cols.shape)}"
            )
        if (
            start_cols.dtype.is_floating_point
            or start_cols.dtype.is_complex
            or start_cols.dtype == torch.bool
        ):
            raise ValueError(f"start_cols must be an integer tensor, got {start_cols.dtype}")
        start_cols = start_cols.to(device=values.device, dtype=torch.long, copy=True)
        # Single fused check: one host-device sync on CUDA instead of two.
        if M > 0 and bool(((start_cols < 0) | (start_cols + L > num_cols)).any()):
            raise ValueError(f"start_cols must lie in [0, {num_cols - L}] for L={L}, N={num_cols}")

        self.values = values
        self.start_cols = start_cols
        # (M, L) column index of every stored element; shared by to_dense and matmul.
        self._col_idx = start_cols[:, None] + torch.arange(L, device=values.device)
        # Transpose structure for the deterministic triton grad_B kernel;
        # built lazily on the first product that needs grad wrt B.
        self._t_cache = None

    def _column_index(self) -> Tensor:
        # `values` may have been swapped to another device in place (e.g. by
        # Module._apply); keep the cached index (and start_cols) where the
        # compute runs.
        if self._col_idx.device != self.values.device:
            self._col_idx = self._col_idx.to(self.values.device)
            self.start_cols = self.start_cols.to(self.values.device)
        return self._col_idx

    def to_dense(self) -> Tensor:
        """Materialize the represented ``M x N`` matrix as a dense tensor."""
        M, N = self.shape
        return self.values.new_zeros(M, N).scatter_(1, self._column_index(), self.values)

    def _matmul_dense(self, other: Tensor) -> Tensor:
        """Compute ``self @ other`` using the compact representation."""
        if other.ndim == 1:  # matrix-vector product, as torch.matmul defines it
            return self._matmul_dense(other.unsqueeze(1)).squeeze(1)
        M, N = self.shape
        if other.ndim != 2 or other.shape[0] != N:
            raise ValueError(
                f"expected other of shape ({N}, K) or ({N},), got {tuple(other.shape)}"
            )
        L = self.values.shape[1]
        if (
            rcs_mm is not None
            and L > 0
            and self.values.is_cuda
            and other.is_cuda
            and self.values.dtype == other.dtype
        ):
            # Fastest path on GPU: fused Triton kernels, differentiable
            # (including double backward via its grad-mode backward branch).
            self._column_index()  # syncs start_cols to the compute device
            if torch.is_grad_enabled() and other.requires_grad:
                if (
                    self._t_cache is None
                    or self._t_cache[0].device != self.values.device
                ):
                    self._t_cache = _build_transpose_cache(self.start_cols, L, N)
            return rcs_mm(self.values, self.start_cols, other, self._t_cache)
        grad_needed = torch.is_grad_enabled() and (
            self.values.requires_grad or other.requires_grad
        )
        if L > 0 and not grad_needed:
            # Inference: cuSPARSE/MKL CSR spmm avoids materializing the
            # (M, L, K) windows. Not used under autograd — CSR matmul is not
            # twice-differentiable and its grad semantics differ from dense.
            idx = self._column_index()
            crow = torch.arange(0, M * L + 1, L, device=idx.device)
            csr = torch.sparse_csr_tensor(
                crow, idx.reshape(-1), self.values.reshape(-1), size=(M, N)
            )
            return csr @ other
        # Row i of the product only needs rows J_i .. J_i + L - 1 of `other`:
        # gather them into (M, L, K) windows and contract with values (M, L).
        windows = other[self._column_index()]
        return torch.bmm(self.values.unsqueeze(1), windows).squeeze(1)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in _MATMUL_FUNCS:
            kw = dict(kwargs)
            out = kw.pop("out", None)
            operands = list(args)
            if not operands and "input" in kw:
                operands.append(kw.pop("input"))
            if len(operands) == 1 and "other" in kw:
                operands.append(kw.pop("other"))
            if len(operands) == 2 and not kw and out is None:
                a, b = operands
                if (
                    isinstance(a, RCSMatrix)
                    and isinstance(b, Tensor)
                    and not isinstance(b, RCSMatrix)
                ):
                    return a._matmul_dense(b)
        # Metadata accessors (shape, dtype, device, ...) work on the
        # storage-less wrapper; data ops fall through to __torch_dispatch__.
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise NotImplementedError(
            f"{func} is not supported on the compact RCS form; "
            "convert with .to_dense() first"
        )

    def __deepcopy__(self, memo: dict) -> "RCSMatrix":
        result = RCSMatrix(
            copy.deepcopy(self.values, memo),
            copy.deepcopy(self.start_cols, memo),
            self.shape[1],
        )
        memo[id(self)] = result
        return result

    def __repr__(self) -> str:
        M, N = self.shape
        return (
            f"RCSMatrix(shape=({M}, {N}), L={self.values.shape[1]}, "
            f"dtype={self.dtype}, device={self.device},\n"
            f"values={self.values},\nstart_cols={self.start_cols})"
        )
