from functools import partial

import torch
import torch.nn as nn

from .modeling_finetune import NeuralTransformer
from .._utils import minmax

PATCH = 100

_RENAMES = {
    "patch_embed_conv.": "patch_embed.",
    "patch_embed_projection.": "patch_proj.",
}


class SigmaPPG(nn.Module):
    out_dim = 360

    def __init__(self):
        super().__init__()
        self.encoder = NeuralTransformer(
            PPG_size=12000, patch_size=PATCH, in_chans=1, out_chans=12, num_classes=0,
            embed_dim=360, depth=18, num_heads=12, mlp_ratio=4.0, qkv_bias=False,
            qk_norm=partial(nn.LayerNorm, eps=1e-6), norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=0.1,
        )
        self.encoder.fc_norm = nn.Identity()

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        n = x.shape[-1] // PATCH
        x = minmax(x[..., : n * PATCH])
        return self.encoder(x.view(x.shape[0], 1, n, PATCH))


def load_sigmappg_from_checkpoint(checkpoint_path, device="cpu"):
    model = SigmaPPG()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    src = {k[len("student."):]: v for k, v in ckpt["model"].items() if k.startswith("student.")}

    state = {}
    for k, v in src.items():
        for old, new in _RENAMES.items():
            if k.startswith(old):
                k = new + k[len(old):]
                break
        state[k] = v
    model.encoder.pos_embed = nn.Parameter(state["pos_embed"])

    missing, unexpected = model.encoder.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if not k.startswith(("mask_token", "lm_head", "patch_embed_seq"))]
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)
    model.to(torch.device(device)).eval()
    print(f"Loaded SIGMA-PPG from: {checkpoint_path}")
    return model
