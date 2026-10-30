import torch
import torch.nn as nn

from .resnet1d import Net1D
from .._utils import zscore


ANYPPG_CFG = dict(
    in_channels=1, base_filters=64, ratio=1.0,
    filter_list=[64, 160, 160, 400, 400, 512], m_blocks_list=[2, 2, 2, 3, 3, 1],
    kernel_size=3, stride=2, groups_width=16, use_bn=True, use_do=True, verbose=False,
)


class AnyPPG(nn.Module):
    out_dim = 512

    def __init__(self):
        super().__init__()
        self.encoder = Net1D(**ANYPPG_CFG)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.encoder(zscore(x))


def load_anyppg_from_checkpoint(checkpoint_path, device="cpu"):
    model = AnyPPG()
    state = torch.load(checkpoint_path, map_location="cpu")
    model.encoder.load_state_dict(state, strict=True)
    model.to(torch.device(device)).eval()
    print(f"Loaded AnyPPG from: {checkpoint_path}")
    return model
