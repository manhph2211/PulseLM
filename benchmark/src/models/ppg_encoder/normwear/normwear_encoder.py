import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .normwear import NormWear
from .._utils import load_weights, resolve_weights

FS = 65


def _preprocess(x: torch.Tensor, fs: int = FS, smooth_c: float = 0.02) -> torch.Tensor:
    t = torch.linspace(-1.0, 1.0, x.shape[-1], device=x.device, dtype=x.dtype)
    slope = (x * t).sum(-1, keepdim=True) / t.square().sum()
    x = x - x.mean(-1, keepdim=True) - slope * t
    sigma = fs * smooth_c
    r = int(math.ceil(4 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / sigma) ** 2)
    k = (k / k.sum()).view(1, 1, -1)
    x = F.conv1d(F.pad(x.unsqueeze(1), (r, r), mode="reflect"), k).squeeze(1)
    return x / (x.abs().mean(-1, keepdim=True) + 1e-6)


class NormWearEncoder(nn.Module):
    out_dim = 768

    def __init__(self):
        super().__init__()
        self.backbone = NormWear(
            img_size=(387, 65), patch_size=(9, 5), in_chans=3, target_len=388, nvar=4,
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, fuse_freq=2,
            is_pretrain=False, mask_prob=0.0, mask_scheme="random", use_cwt=True,
            comb_freq=False, include_msitf=False,
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        x = _preprocess(x.float()).unsqueeze(1)
        spec = self.backbone.calc_cwt(x).to(self.backbone.cls_token.dtype)
        out = self.backbone.get_signal_embedding(spec, device=x.device)
        return out.mean(dim=2).squeeze(1)


def load_normwear_from_checkpoint(checkpoint_path="mosaic-laboratory/normwear", device="cpu"):
    model = NormWearEncoder()
    state = load_weights(resolve_weights(checkpoint_path, "model.safetensors"))
    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    pretrain_only = ("decoder", "mask_token", "spatial_recon", "temporal_recon", "msitf")
    unexpected = [k for k in unexpected if not k.startswith(pretrain_only)]
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)
    model.to(torch.device(device)).eval()
    print(f"Loaded NormWear from: {checkpoint_path}")
    return model
