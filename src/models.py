"""DCGAN generator and discriminator for 64x64 RGB images (Radford et al., 2016)."""
import torch
from torch import nn
from torch.nn.utils import spectral_norm

Z_DIM = 100


class Generator(nn.Module):
    def __init__(self, z_dim: int = Z_DIM, width: int = 64):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            # z: (N, z_dim, 1, 1) -> (N, 8w, 4, 4)
            nn.ConvTranspose2d(z_dim, w * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(w * 8),
            nn.ReLU(True),
            # -> (N, 4w, 8, 8)
            nn.ConvTranspose2d(w * 8, w * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(w * 4),
            nn.ReLU(True),
            # -> (N, 2w, 16, 16)
            nn.ConvTranspose2d(w * 4, w * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(w * 2),
            nn.ReLU(True),
            # -> (N, w, 32, 32)
            nn.ConvTranspose2d(w * 2, w, 4, 2, 1, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(True),
            # -> (N, 3, 64, 64)
            nn.ConvTranspose2d(w, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.view(z.size(0), -1, 1, 1))


class Discriminator(nn.Module):
    """Outputs one raw logit per image. `spectral` wraps every conv in spectral norm
    and changes nothing else, so it can be compared one-to-one with the base.
    `batchnorm=False` drops BN from D, as in SN-GAN (Miyato et al., 2018)."""

    def __init__(self, width: int = 64, spectral: bool = False, batchnorm: bool = True):
        super().__init__()
        w = width
        sn = spectral_norm if spectral else (lambda m: m)
        norm = nn.BatchNorm2d if batchnorm else (lambda c: nn.Identity())
        self.net = nn.Sequential(
            # (N, 3, 64, 64) -> (N, w, 32, 32); no BN on the first layer, as in DCGAN
            sn(nn.Conv2d(3, w, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, True),
            # -> (N, 2w, 16, 16)
            sn(nn.Conv2d(w, w * 2, 4, 2, 1, bias=False)),
            norm(w * 2),
            nn.LeakyReLU(0.2, True),
            # -> (N, 4w, 8, 8)
            sn(nn.Conv2d(w * 2, w * 4, 4, 2, 1, bias=False)),
            norm(w * 4),
            nn.LeakyReLU(0.2, True),
            # -> (N, 8w, 4, 4)
            sn(nn.Conv2d(w * 4, w * 8, 4, 2, 1, bias=False)),
            norm(w * 8),
            nn.LeakyReLU(0.2, True),
            # -> (N, 1, 1, 1)
            sn(nn.Conv2d(w * 8, 1, 4, 1, 0, bias=False)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def init_weights(m: nn.Module) -> None:
    """DCGAN init: conv ~ N(0, 0.02), BN gamma ~ N(1, 0.02), beta = 0."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # Spectral norm stores the raw weight as weight_orig.
        weight = getattr(m, "weight_orig", m.weight)
        nn.init.normal_(weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
