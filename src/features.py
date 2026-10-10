"""Pretrained Inception-v3 pool features, shared by the metrics and the nearest-neighbor test."""
import torch
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class FeatureExtractor:
    """Maps images in [-1, 1] to 2048-d Inception pool features (ImageNet weights)."""

    def __init__(self, device: torch.device):
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, batch: int = 200) -> torch.Tensor:
        out = []
        for i in range(0, x.size(0), batch):
            b = x[i:i + batch].to(self.device)
            b = (b + 1) / 2  # [-1, 1] -> [0, 1]
            b = F.interpolate(b, size=299, mode="bilinear", align_corners=False)
            b = (b - _MEAN.to(self.device)) / _STD.to(self.device)
            out.append(self.model(b).float().cpu())
        return torch.cat(out)
