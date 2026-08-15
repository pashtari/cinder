# CINDER

Spatially-conditioned Implicit Neural Representation for Dense prediction.

CINDER explores conditional INR-based image segmentation where encoder features are
fused with spatial coordinates (rather than modulating INR weights).

## Features

- **Coordinate-Conditioning**: encoder features are pooled into a conditioning
vector (via a learnable bilinear map) that modulates a coordinate-driven INR.
- **Conditional INR Decoder**: ships with `ConditionalFUTON`, a Fourier tensor
network whose CP-decomposed weights are contracted with the condition vector.
- **Flexible Sampling**: supports coordinate subsampling during training for efficiency.


## Installation

**Prerequisites:** Python ≥ 3.10, PyTorch ≥ 2.10

```bash
git clone https://github.com/pashtari/cinder.git
cd cinder
pip install -e .          # core library
pip install -e ".[train]" # + training dependencies (Ignite, Hydra, TensorBoard)
```


## Quick Start


```python
import torch
import cinder as cd

x = torch.randn(8, 3, 224, 224)

encoder = cd.Encoder(
    model_name="tiny_vit_21m_384.dist_in22k_ft_in1k", pretrained=True, normalize=True
)
cinr = cd.ConditionalFUTON(
    in_features=2,  # 2D coordinates
    out_features=1,
    num_components=256,
    cond_dim=256,
    rank=256,
)
model = cd.CINDER(
    encoder=encoder,
    cinr=cinr,
    in_size=(224, 224),
)

y = model(x)  # (8, 1, 224, 224)
```


## License

This project is licensed under the MIT License.
