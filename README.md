# CINDER

Conditioned Implicit Neural DecodeR for dense prediction.

CINDER segments an image by decoding every pixel coordinate with an implicit
neural representation (INR) conditioned on the feature maps of an image encoder.
For query coordinates `x` and encoder maps `z`, it predicts
`y = INR_theta(z)(phi(x, z))` with a list of modulators, each wrapping the ones
before it:

- **Input modulators** transform the INR input `phi(x, z)`: `FUTONGate` encodes
  each coordinate with a Fourier tensor network and gates the encoding with the
  encoder maps sampled at that coordinate.
- **Weight modulators** change the INR parameters `theta(z)`:
  `WeightDisplacement` predicts, from every position of the final encoder map, an
  additive displacement of the INR's weight matrices for each image.

The INR can be `MLP`, `SIREN`, `FINER`, `Gauss`, `WIRE` (real-valued), `RFF` or
`FUTON`, and the encoder any [timm](https://github.com/huggingface/pytorch-image-models)
backbone with dense feature maps.

## Installation

Python 3.10+ and PyTorch 2.1+ are required.

```bash
git clone https://github.com/pashtari/cinder.git
cd cinder
pip install -e ".[train,test]"  # or `pip install -e .` for the library alone
```

## Quick start

```python
from functools import partial

import torch
import cinder as cd

model = cd.CINDER(
    in_channels=3,
    out_channels=1,
    in_size=(64, 64),
    encoder=partial(
        cd.TimmEncoder,
        model_name="resnet18",
        pretrained=False,
        out_indices=(-2, -1),
    ),
    inr=partial(cd.MLP, hidden_features=32, hidden_layers=1),
    modulator=partial(
        cd.ListModulators,
        modulators=[  # innermost first
            ("displacement", {"conditioner": "mlp"}),
            (
                "futon",
                {
                    "basis": ("cosine", {"num_components": 32}),
                    "combiner": ("cp", {"rank": 32}),
                    "fusion": "sum",
                },
            ),
        ],
    ),
)

model.eval()
with torch.no_grad():
    logits = model(torch.randn(2, 3, 64, 64))  # (2, 1, 64, 64)
```

For one mechanism, pass its spec directly, such as `modulator="displacement"` or
`modulator=("futon", {...})`. Components are passed as factories:
CINDER builds the encoder, reads the shapes of its maps, and sizes the modulators
and INR to match. The example runs offline; set `pretrained=True` for pretrained
encoder weights.

[`notebooks/glas_segmentation.ipynb`](notebooks/glas_segmentation.ipynb) walks
through a full experiment: it trains a FUTON gate over a ReLU MLP on GlaS, with
and without weight modulation, scores both on Test A and Test B, and decodes
the INR off the pixel grid (`pip install -e ".[notebook]"`).

## Training and evaluation

Experiments are configured with [Hydra](https://hydra.cc) in `configs/`:

| Group | Options |
|---|---|
| `dataset` | `glas`, `fives`, `ade20k`; each selects the trainer config of the same name |
| `model` | `futongate_relu`, `futongate_weight_relu`, `futongate_weight_siren`, `weight_relu`, `weight_futon`, `weight_siren` |
| `metric` | `dice`, `iou`, `hd95`, `object_f1`, `object_dice`, `object_hausdorff` |
| `inferer` | `default` (sliding window), `tta` (multi-scale and flip) |

`scripts/train.sh` and `scripts/eval.sh` run on the CPU, one GPU, or several GPUs
with torchrun (`--gpus`); all other arguments are Hydra overrides. Run either
script with `--help` for more examples.

```bash
scripts/train.sh --gpus=1 --dataset=glas --path.dataset_dir=/path/to/GlaS
scripts/eval.sh --gpus=1 --dataset=glas --dataset.test_split=testA \
    --metric='[object_f1,object_dice,object_hausdorff]' \
    --path.dataset_dir=/path/to/GlaS --handler.checkpoint.load_from=/path/to/checkpoint.pt
```

Each run writes its config, log, checkpoints and TensorBoard files to
`logs/<dataset>/<tag>/<model>/<timestamp>/`. The dataset layouts are described in
`cinder/datasets/`. GlaS is reported on Test A and Test B separately with the
object metrics, and ADE20K with `metric=iou`, single-scale and `inferer=tta`
separately. The [HPC guide](scripts/hpc/README.md) covers cluster jobs.

The model configs define six variants:

| Model config | Input modulation | Weight modulation | INR |
|---|---|---|---|
| `futongate_relu` | `FUTONGate` | — | ReLU `MLP` |
| `futongate_weight_relu` | `FUTONGate` | `WeightDisplacement` | ReLU `MLP` |
| `futongate_weight_siren` | `FUTONGate` | `WeightDisplacement` | `SIREN` |
| `weight_relu` | — | `WeightDisplacement` | ReLU `MLP` |
| `weight_futon` | — | `WeightDisplacement` | `FUTON` |
| `weight_siren` | — | `WeightDisplacement` | `SIREN` |

The gate reads the encoder's last four stages; weight modulation reads the final
stage.

## Design

**Encoders.** `TimmEncoder` returns one `(B, C_i, H_i, W_i)` map per selected
stage, at its native resolution, for CNNs and transformers alike; `out_indices`,
`embed_dims` and `feature_reductions` describe them. Channels-last outputs are
permuted, and timm's `features_only` adapters return transformer tokens as patch
grids. Fixed-size transformers need an `img_size` that matches CINDER's
`in_size`, and patch dropout must stay 0. Input normalization is a learnable
affine initialized from the backbone's pretrained mean and std, and `normalize`
adds a LayerNorm to each output map.

**Modulators.** Every modulator subclasses `BaseModulator`: it is built from
`(inr, in_features, out_features, cond_shape)` and predicts with
`forward(coords, conds) -> (B, *, out_features)`. Coordinates are
`(B, *, in_features)`, or `(1, *, in_features)` when the batch shares them, and
`cond_shape` holds the `(C, *grid)` shape of each map without the batch axis,
which CINDER infers from the encoder. `inr` is the model a modulator wraps: a
plain INR or another modulator. `ListModulators` nests its list innermost first,
so `[displacement, gate]` is the gate wrapping the displacement wrapping the
INR: the gate hands its gated features to the model it wraps as that model's
coordinates, and the displacement runs its model once per image under
`torch.func.vmap` with displaced matrices (a `FUTON` INR with a local basis then
needs `sparse: false`). A new modulator subclasses `BaseModulator`, builds its
model with `build_inr`, and can be added to `MODULATORS`. A model config names a
single modulator directly, as in `futongate_relu`, or `cinder.ListModulators`
with a list, as in `futongate_weight_relu`.

**Specs.** Components such as the INR, basis, combiner, fusion and conditioner
accept a registry key, a `(key, parameters)` pair, a factory or a module.

## Tests

```bash
python -m pytest
```

The tests build small models without downloading weights, and run the CUDA
checks when a GPU is available.

## License

MIT. See [LICENSE](LICENSE).
