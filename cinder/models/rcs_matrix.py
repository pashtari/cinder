"""Row-contiguous sparse matrices with compact dense multiplication.

An ``(M, N)`` matrix stores at most L consecutive entries per row as
``values (M, L)`` and ``start_cols (M,)``. CUDA products use Triton kernels
with a deterministic first-order backward; higher-order derivatives use
differentiable PyTorch operations. Fallback products use CSR where supported or
a differentiable gather-and-contract path.

Triton unrolls the L taps at compile time, so large segments increase
compilation cost. Accumulation uses float32, or float64 for double inputs.
"""

import copy
from typing import Any

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

__all__ = ["RCSMatrix"]


# Support both AMP APIs: torch.amp added these helpers in PyTorch 2.4.
if hasattr(torch.amp, "custom_fwd"):
    _custom_fwd = torch.amp.custom_fwd(device_type="cuda")
    _custom_bwd = torch.amp.custom_bwd(device_type="cuda")

    def _autocast_enabled() -> bool:
        return torch.is_autocast_enabled("cuda")

    def _autocast_dtype() -> torch.dtype:
        return torch.get_autocast_dtype("cuda")

    def _cpu_autocast_enabled() -> bool:
        return torch.is_autocast_enabled("cpu")

else:  # torch < 2.4
    _custom_fwd = torch.cuda.amp.custom_fwd
    _custom_bwd = torch.cuda.amp.custom_bwd
    _autocast_enabled = torch.is_autocast_enabled
    _autocast_dtype = torch.get_autocast_gpu_dtype
    _cpu_autocast_enabled = torch.is_autocast_cpu_enabled

_MATMUL_FUNCS = {
    torch.matmul,
    torch.mm,
    Tensor.matmul,
    Tensor.mm,
    Tensor.__matmul__,
}


def _build_transpose_cache(
    start_cols: Tensor, num_taps: int, num_cols: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Group stored entries by column for deterministic gradient reduction.

    Returns row pointers ``(N+1,)`` and source row/tap indices ``(M*L,)``.
    The stable sort fixes summation order. Requires ``num_taps > 0``.
    """
    device = start_cols.device
    starts = start_cols.to(torch.long).reshape(-1)
    columns = (
        starts[:, None] + torch.arange(num_taps, device=device, dtype=torch.long)
    ).reshape(-1)
    order = torch.argsort(columns, stable=True)
    row_offsets = torch.searchsorted(
        columns[order], torch.arange(num_cols + 1, device=device, dtype=torch.long)
    )
    return (
        row_offsets.contiguous(),
        (order // num_taps).to(torch.int32).contiguous(),
        (order % num_taps).to(torch.int32).contiguous(),
    )


if triton is not None:
    # Tune tile shape with low first-call overhead. Forward replay only overwrites
    # its output, so autotuning needs no reset; backward kernels are not autotuned.
    _FWD_CONFIGS = [
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 32}, num_warps=4),
    ]

    @triton.autotune(configs=_FWD_CONFIGS, key=["K", "L"])
    @triton.jit
    def _rcs_fwd_kernel(
        g_ptr,
        j_ptr,
        b_ptr,
        out_ptr,
        M,
        K,
        stride_gm,
        stride_gl,
        stride_bn,
        stride_bk,
        stride_om,
        stride_ok,
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
        for tap in tl.static_range(L):
            g = tl.load(
                g_ptr + rm * stride_gm + tap * stride_gl, mask=mask_m, other=0.0
            )
            b = tl.load(
                b_ptr + (j + tap)[:, None] * stride_bn + rk[None, :] * stride_bk,
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
        go_ptr,
        j_ptr,
        b_ptr,
        gg_ptr,
        M,
        K,
        stride_gom,
        stride_gok,
        stride_bn,
        stride_bk,
        stride_ggm,
        stride_ggl,
        L: tl.constexpr,
        L_P2: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """Reuse each grad_out tile across all L row-dot-product accumulators."""
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
        rowptr_ptr,
        srci_ptr,
        srcl_ptr,
        g_ptr,
        go_ptr,
        gb_ptr,
        K,
        stride_gm,
        stride_gl,
        stride_gom,
        stride_gok,
        stride_gbn,
        stride_gbk,
        BLOCK_E: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """Reduce each column's contributors in fixed order for determinism."""
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
            tap = tl.load(srcl_ptr + re, mask=mask_e, other=0).to(tl.int64)
            g = tl.load(
                g_ptr + i * stride_gm + tap * stride_gl, mask=mask_e, other=0.0
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

    def _acc_tl_dtype(dtype: torch.dtype) -> tl.dtype:
        return tl.float64 if dtype == torch.float64 else tl.float32

    def _launch_forward(values: Tensor, start_cols: Tensor, other: Tensor) -> Tensor:
        num_rows, num_taps = values.shape
        out_features = other.shape[1]
        out = torch.empty(
            num_rows, out_features, dtype=values.dtype, device=values.device
        )
        if num_rows == 0 or out_features == 0:
            return out
        if num_taps == 0:
            return out.zero_()

        def grid(meta):
            return (
                triton.cdiv(num_rows, meta["BLOCK_M"]),
                triton.cdiv(out_features, meta["BLOCK_K"]),
            )

        _rcs_fwd_kernel[grid](
            values,
            start_cols,
            other,
            out,
            num_rows,
            out_features,
            values.stride(0),
            values.stride(1),
            other.stride(0),
            other.stride(1),
            out.stride(0),
            out.stride(1),
            L=num_taps,
            ACC_DTYPE=_acc_tl_dtype(values.dtype),
        )
        return out

    def _launch_grad_values(
        grad_out: Tensor,
        start_cols: Tensor,
        other: Tensor,
        num_rows: int,
        num_taps: int,
    ) -> Tensor:
        out_features = grad_out.shape[1]
        grad_values = torch.empty(
            num_rows, num_taps, dtype=grad_out.dtype, device=grad_out.device
        )
        if num_rows == 0 or num_taps == 0:
            return grad_values
        if out_features == 0:
            return grad_values.zero_()
        block_k = min(128, triton.next_power_of_2(out_features))
        grid = (triton.cdiv(num_rows, 16),)
        _rcs_bwd_g_kernel[grid](
            grad_out,
            start_cols,
            other,
            grad_values,
            num_rows,
            out_features,
            grad_out.stride(0),
            grad_out.stride(1),
            other.stride(0),
            other.stride(1),
            grad_values.stride(0),
            grad_values.stride(1),
            L=num_taps,
            L_P2=triton.next_power_of_2(num_taps),
            BLOCK_M=16,
            BLOCK_K=block_k,
            ACC_DTYPE=_acc_tl_dtype(grad_out.dtype),
            num_warps=4,
        )
        return grad_values

    def _launch_grad_other(
        values: Tensor,
        cache: tuple[Tensor, Tensor, Tensor] | None,
        grad_out: Tensor,
        num_cols: int,
    ) -> Tensor:
        """Reduce the dense operand's gradient in float32/float64; the caller casts."""
        num_rows, num_taps = values.shape
        out_features = grad_out.shape[1]
        acc_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
        if num_rows == 0 or out_features == 0 or num_taps == 0 or num_cols == 0:
            return torch.zeros(
                num_cols, out_features, dtype=acc_dtype, device=values.device
            )
        row_offsets, source_rows, source_taps = cache
        grad_other = torch.empty(
            num_cols, out_features, dtype=acc_dtype, device=values.device
        )
        block_k = min(128, triton.next_power_of_2(out_features))
        grid = (num_cols, triton.cdiv(out_features, block_k))
        _rcs_bwd_b_kernel[grid](
            row_offsets,
            source_rows,
            source_taps,
            values,
            grad_out,
            grad_other,
            out_features,
            values.stride(0),
            values.stride(1),
            grad_out.stride(0),
            grad_out.stride(1),
            grad_other.stride(0),
            grad_other.stride(1),
            BLOCK_E=32,
            BLOCK_K=block_k,
            ACC_DTYPE=_acc_tl_dtype(acc_dtype),
            num_warps=4,
        )
        return grad_other

    class RCSMatmulFn(torch.autograd.Function):
        """Triton product with a differentiable path for higher-order gradients.

        ``cache`` groups entries by column for a deterministic dense gradient.
        RCSMatrix reuses it across products; otherwise it is built on demand.
        """

        @staticmethod
        @_custom_fwd
        def forward(
            ctx,
            values: Tensor,
            start_cols: Tensor,
            other: Tensor,
            cache: tuple[Tensor, Tensor, Tensor] | None = None,
        ) -> Tensor:
            if not (values.is_cuda and other.is_cuda and start_cols.is_cuda):
                raise RuntimeError("triton RCS matmul requires CUDA tensors")

            compute_dtype = torch.promote_types(values.dtype, other.dtype)
            # Match matmul autocast while preserving explicit low/double precision.
            if compute_dtype == torch.float32 and _autocast_enabled():
                compute_dtype = _autocast_dtype()

            start_cols = start_cols.to(torch.long).contiguous()
            out = _launch_forward(
                values.to(compute_dtype), start_cols, other.to(compute_dtype)
            )

            # Keep original operands so backward can differentiate through casts.
            ctx.save_for_backward(values, start_cols, other)
            ctx.compute_dtype = compute_dtype
            ctx.cache = cache
            return out

        @staticmethod
        @_custom_bwd
        def backward(
            ctx, grad_out: Tensor
        ) -> tuple[Tensor | None, None, Tensor | None, None]:
            values, start_cols, other = ctx.saved_tensors
            needs_values_grad, _, needs_other_grad = ctx.needs_input_grad[:3]
            num_rows, num_taps = values.shape
            num_cols = other.shape[0]
            grad_values = grad_other = None

            if torch.is_grad_enabled():
                # Build a graph through backward for higher-order derivatives.
                column_indices = start_cols[:, None] + torch.arange(
                    num_taps, device=start_cols.device
                )
                cast_values = values.to(grad_out.dtype)
                cast_other = other.to(grad_out.dtype)
                if needs_values_grad:
                    grad_values = (
                        (grad_out.unsqueeze(1) * cast_other[column_indices])
                        .sum(-1)
                        .to(values.dtype)
                    )
                if needs_other_grad:
                    contributions = (
                        cast_values.unsqueeze(-1) * grad_out.unsqueeze(1)
                    ).reshape(num_rows * num_taps, grad_out.shape[1])
                    grad_other = (
                        torch.zeros_like(cast_other)
                        .index_add_(0, column_indices.reshape(-1), contributions)
                        .to(other.dtype)
                    )
                return grad_values, None, grad_other, None

            compute_dtype = ctx.compute_dtype
            cast_grad = grad_out.to(compute_dtype)
            if needs_values_grad:
                grad_values = _launch_grad_values(
                    cast_grad, start_cols, other.to(compute_dtype), num_rows, num_taps
                )
                grad_values = grad_values.to(values.dtype)
            if needs_other_grad:
                cache = ctx.cache
                if cache is None and num_taps > 0:
                    cache = _build_transpose_cache(start_cols, num_taps, num_cols)
                grad_other = _launch_grad_other(
                    values.to(compute_dtype), cache, cast_grad, num_cols
                )
                grad_other = grad_other.to(other.dtype)
            return grad_values, None, grad_other, None

    def rcs_mm(
        values: Tensor,
        start_cols: Tensor,
        other: Tensor,
        cache: tuple[Tensor, Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Multiply the compact matrix by a dense ``(N, K)`` matrix or ``(N,)`` vector."""
        if other.ndim == 1:
            return RCSMatmulFn.apply(
                values, start_cols, other.unsqueeze(1), cache
            ).squeeze(1)
        return RCSMatmulFn.apply(values, start_cols, other, cache)

else:
    rcs_mm = None


class RCSMatrix(torch.Tensor):
    """Sparse matrix with one fixed-width segment of stored values per row.

    ``matrix @ other`` accepts a dense matrix or vector, with gradients to
    both operands and support for higher-order derivatives. Other tensor
    operations require :meth:`to_dense`. Deep copying preserves compact storage.

    Args:
        values: Segment values of shape ``(M, L)``, padded with zeros as needed.
        start_cols: Integer start column per row, shape ``(M,)``, with
            ``0 <= start_cols[i] <= num_cols - L``. Copied at construction.
        num_cols: Number of columns in the represented matrix.

    If ``values`` moves to another device in place, computation follows it,
    but the wrapper's ``device`` metadata retains its construction-time value.
    """

    values: Tensor
    start_cols: Tensor

    @staticmethod
    def __new__(cls, values: Tensor, start_cols: Tensor, num_cols: int) -> "RCSMatrix":
        if values.ndim != 2:
            raise ValueError(
                f"values must have shape (M, L), got {tuple(values.shape)}"
            )
        if num_cols < 0:
            raise ValueError(f"num_cols must be non-negative, got {num_cols}")
        return torch.Tensor._make_wrapper_subclass(
            cls,
            (values.shape[0], num_cols),
            dtype=values.dtype,
            device=values.device,
            requires_grad=values.requires_grad,
        )

    def __init__(self, values: Tensor, start_cols: Tensor, num_cols: int) -> None:
        num_rows, num_taps = values.shape
        if start_cols.shape != (num_rows,):
            raise ValueError(
                f"start_cols must have shape ({num_rows},), got {tuple(start_cols.shape)}"
            )
        if (
            start_cols.dtype.is_floating_point
            or start_cols.dtype.is_complex
            or start_cols.dtype == torch.bool
        ):
            raise ValueError(
                f"start_cols must be an integer tensor, got {start_cols.dtype}"
            )
        start_cols = start_cols.to(device=values.device, dtype=torch.long, copy=True)
        # Combine bounds checks to avoid a second host-device sync on CUDA.
        if num_rows > 0 and bool(
            ((start_cols < 0) | (start_cols + num_taps > num_cols)).any()
        ):
            raise ValueError(
                f"start_cols must lie in [0, {num_cols - num_taps}] for L={num_taps}, N={num_cols}"
            )

        self.values = values
        self.start_cols = start_cols
        self._col_idx = start_cols[:, None] + torch.arange(
            num_taps, device=values.device
        )
        # Build lazily when a product first needs the dense operand's gradient.
        self._transpose_cache: tuple[Tensor, Tensor, Tensor] | None = None

    def _column_index(self) -> Tensor:
        # Values may move in place through Module._apply; move cached indices too.
        if self._col_idx.device != self.values.device:
            self._col_idx = self._col_idx.to(self.values.device)
            self.start_cols = self.start_cols.to(self.values.device)
        return self._col_idx

    def to_dense(self) -> Tensor:
        """Materialize the represented ``M x N`` matrix as a dense tensor."""
        num_rows, num_cols = self.shape
        return self.values.new_zeros(num_rows, num_cols).scatter_(
            1, self._column_index(), self.values
        )

    def _matmul_dense(self, other: Tensor) -> Tensor:
        """Compute ``self @ other`` using the compact representation."""
        if other.ndim == 1:
            return self._matmul_dense(other.unsqueeze(1)).squeeze(1)
        num_rows, num_cols = self.shape
        if other.ndim != 2 or other.shape[0] != num_cols:
            raise ValueError(
                f"expected other of shape ({num_cols}, K) or ({num_cols},), got {tuple(other.shape)}"
            )
        if self.values.device != other.device:
            raise ValueError(
                "RCS values and the dense operand must be on the same device"
            )
        num_taps = self.values.shape[1]
        if (
            rcs_mm is not None
            and num_taps > 0
            and self.values.is_cuda
            and other.is_cuda
            and self.values.dtype == other.dtype
            and self.values.dtype
            in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        ):
            self._column_index()  # syncs start_cols to the compute device
            if torch.is_grad_enabled() and other.requires_grad:
                if (
                    self._transpose_cache is None
                    or self._transpose_cache[0].device != self.values.device
                ):
                    self._transpose_cache = _build_transpose_cache(
                        self.start_cols, num_taps, num_cols
                    )
            return rcs_mm(self.values, self.start_cols, other, self._transpose_cache)
        grad_needed = torch.is_grad_enabled() and (
            self.values.requires_grad or other.requires_grad
        )
        # CPU CSR kernels lack half/bfloat16 support, including under autocast.
        cpu_csr_supported = (
            self.values.dtype
            in (torch.float32, torch.float64, torch.complex64, torch.complex128)
            and not _cpu_autocast_enabled()
        )
        if (
            num_taps > 0
            and not grad_needed
            and (self.values.is_cuda or cpu_csr_supported)
        ):
            # CSR avoids gathered windows, but lacks the higher-order gradients
            # needed by the autograd path.
            column_indices = self._column_index()
            row_offsets = torch.arange(
                0, num_rows * num_taps + 1, num_taps, device=column_indices.device
            )
            csr = torch.sparse_csr_tensor(
                row_offsets,
                column_indices.reshape(-1),
                self.values.reshape(-1),
                size=(num_rows, num_cols),
            )
            return csr @ other
        # Each output row needs only L rows of the dense operand.
        windows = other[self._column_index()]
        return torch.bmm(self.values.unsqueeze(1), windows).squeeze(1)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in _MATMUL_FUNCS:
            remaining_kwargs = dict(kwargs)
            out = remaining_kwargs.pop("out", None)
            operands = list(args)
            if not operands and "input" in remaining_kwargs:
                operands.append(remaining_kwargs.pop("input"))
            if len(operands) == 1 and "other" in remaining_kwargs:
                operands.append(remaining_kwargs.pop("other"))
            if len(operands) == 2 and not remaining_kwargs and out is None:
                matrix, other = operands
                if (
                    isinstance(matrix, RCSMatrix)
                    and isinstance(other, Tensor)
                    and not isinstance(other, RCSMatrix)
                ):
                    return matrix._matmul_dense(other)
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

    def __deepcopy__(self, memo: dict[int, Any]) -> "RCSMatrix":
        result = RCSMatrix(
            copy.deepcopy(self.values, memo),
            copy.deepcopy(self.start_cols, memo),
            self.shape[1],
        )
        memo[id(self)] = result
        return result

    def __repr__(self) -> str:
        num_rows, num_cols = self.shape
        return (
            f"RCSMatrix(shape=({num_rows}, {num_cols}), L={self.values.shape[1]}, "
            f"dtype={self.dtype}, device={self.device},\n"
            f"values={self.values},\nstart_cols={self.start_cols})"
        )
