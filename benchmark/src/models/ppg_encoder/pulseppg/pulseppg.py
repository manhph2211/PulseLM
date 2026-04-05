import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv1dPadSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels,
                              kernel_size=self.kernel_size, stride=self.stride, groups=self.groups)

    def forward(self, x):
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        x = F.pad(x, (p // 2, p - p // 2), "constant", 0)
        return self.conv(x)


class MyMaxPool1dPadSame(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        x = F.pad(x, (p // 2, p - p // 2), "constant", 0)
        return self.max_pool(x)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do,
                 is_first_block=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride if downsample else 1
        self.downsample = downsample
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, stride=self.stride, groups=groups)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(in_channels=out_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, stride=1, groups=groups)
        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        identity = x
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        if self.downsample:
            identity = self.max_pool(identity)
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)
        out += identity
        return out


class Net(nn.Module):
    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block,
                 finalpool=None, downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True):
        super().__init__()
        self.n_block = n_block
        self.use_bn = use_bn
        self.use_do = use_do
        self.downsample_gap = downsample_gap
        self.increasefilter_gap = increasefilter_gap
        self.finalpool = finalpool

        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters,
                                                kernel_size=kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()

        self.basicblock_list = nn.ModuleList()
        out_channels = base_filters
        for i_block in range(self.n_block):
            is_first_block = (i_block == 0)
            downsample = (i_block % self.downsample_gap == 1)
            if is_first_block:
                in_c = base_filters
                out_c = in_c
            else:
                in_c = int(base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap))
                out_c = in_c * 2 if (i_block % self.increasefilter_gap == 0) else in_c
            out_channels = out_c
            self.basicblock_list.append(BasicBlock(
                in_channels=in_c, out_channels=out_c, kernel_size=kernel_size,
                stride=stride, groups=groups, downsample=downsample,
                use_bn=use_bn, use_do=use_do, is_first_block=is_first_block))

        self.instnorm = nn.InstanceNorm1d(in_channels)

    def forward(self, x):
        out = self.instnorm(x)
        out = self.first_block_conv(out)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        for block in self.basicblock_list:
            out = block(out)
        if self.finalpool == "avg":
            out = torch.mean(out, dim=-1)
        elif self.finalpool == "max":
            out = torch.max(out, dim=-1)[0]
        return out


class PulsePPG(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Net(in_channels=1, base_filters=128, kernel_size=11, stride=2,
                           groups=1, n_block=12, finalpool="max")

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.encoder(x)


def load_pulseppg_from_checkpoint(checkpoint_path, device="cpu"):
    device = torch.device(device)
    model = PulsePPG().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    net_state = state["net"] if isinstance(state, dict) and "net" in state else state
    missing, unexpected = model.encoder.load_state_dict(net_state, strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)
    model.eval()
    print(f"Loaded PulsePPG from: {checkpoint_path}")
    return model


if __name__ == "__main__":
    import os

    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "checkpoints", "pulseppg", "checkpoint_best.pkl")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(ckpt_path):
        model = load_pulseppg_from_checkpoint(ckpt_path, device=device)
    else:
        print(f"Checkpoint not found: {ckpt_path}, using random weights.")
        model = PulsePPG().to(device).eval()

    x = torch.randn(2, 1250, device=device)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 512), f"Unexpected shape: {out.shape}"
    print(f"Input={x.shape} Output={out.shape} OK")
