import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .model_utils import load_model_without_module_prefix
except ImportError:
    from model_utils import load_model_without_module_prefix


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
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.downsample = downsample
        self.stride = stride if downsample else 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, stride=self.stride, groups=self.groups)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(in_channels=out_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, stride=1, groups=self.groups)
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


class ResNet1D(nn.Module):
    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes,
                 downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, verbose=False,
                 use_mt_regression=False, use_projection=False):
        super().__init__()
        self.verbose = verbose
        self.n_block = n_block
        self.use_bn = use_bn
        self.use_do = use_do
        self.use_mt_regression = use_mt_regression
        self.use_projection = use_projection
        self.downsample_gap = downsample_gap
        self.increasefilter_gap = increasefilter_gap

        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters,
                                                kernel_size=kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters

        self.basicblock_list = nn.ModuleList()
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

        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.dense = nn.Linear(out_channels, n_classes)

        if self.use_projection:
            self.projector = nn.Sequential(
                nn.Linear(out_channels, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 128))

        if self.use_mt_regression:
            self.mt_regression = nn.Sequential(
                nn.Linear(n_classes, n_classes // 2), nn.BatchNorm1d(n_classes // 2),
                nn.Linear(n_classes // 2, n_classes // 4), nn.BatchNorm1d(n_classes // 4),
                nn.Linear(n_classes // 4, 1))

    def forward(self, x):
        out = self.first_block_conv(x)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        for block in self.basicblock_list:
            out = block(out)
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out_emb = out.mean(-1)
        if self.use_projection:
            out = self.projector(out_emb)
        else:
            out = self.dense(out_emb)
        if self.use_mt_regression:
            return out, self.mt_regression(out_emb), out_emb
        return out, out_emb


class ResNet1DMoE(nn.Module):
    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes,
                 n_experts=2, downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True,
                 verbose=False, use_projection=False):
        super().__init__()
        self.verbose = verbose
        self.n_block = n_block
        self.use_bn = use_bn
        self.use_do = use_do
        self.use_projection = use_projection
        self.downsample_gap = downsample_gap
        self.increasefilter_gap = increasefilter_gap
        self.n_experts = n_experts

        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters,
                                                kernel_size=kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters

        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            is_first_block = (i_block == 0)
            downsample = (i_block % self.downsample_gap == 1)
            in_c = base_filters if is_first_block else int(base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap))
            out_c = in_c * 2 if (i_block % self.increasefilter_gap == 0 and i_block != 0) else in_c
            out_channels = out_c
            self.basicblock_list.append(BasicBlock(
                in_channels=in_c, out_channels=out_c, kernel_size=kernel_size,
                stride=stride, groups=groups, downsample=downsample,
                use_bn=use_bn, use_do=use_do, is_first_block=is_first_block))

        if self.use_projection:
            self.projector = nn.Sequential(
                nn.Linear(out_channels, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 128))

        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.dense = nn.Linear(out_channels, n_classes)

        self.expert_layers_1 = nn.ModuleList([
            nn.Sequential(nn.Linear(out_channels, out_channels // 2), nn.ReLU(),
                          nn.Linear(out_channels // 2, 1))
            for _ in range(self.n_experts)])
        self.gating_network_1 = nn.Sequential(nn.Linear(out_channels, self.n_experts), nn.Softmax(dim=1))

        self.expert_layers_2 = nn.ModuleList([
            nn.Sequential(nn.Linear(out_channels, out_channels // 2), nn.ReLU(),
                          nn.Dropout(0.3), nn.Linear(out_channels // 2, 1))
            for _ in range(self.n_experts)])
        self.gating_network_2 = nn.Sequential(nn.Linear(out_channels, self.n_experts), nn.Softmax(dim=1))

    def forward(self, x):
        out = self.first_block_conv(x)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        for block in self.basicblock_list:
            out = block(out)
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        out_class = self.projector(out) if self.use_projection else self.dense(out)
        gate1 = self.gating_network_1(out)
        out_moe1 = torch.sum(gate1.unsqueeze(2) * torch.stack([e(out) for e in self.expert_layers_1], dim=1), dim=1)
        gate2 = self.gating_network_2(out)
        out_moe2 = torch.sum(gate2.unsqueeze(2) * torch.stack([e(out) for e in self.expert_layers_2], dim=1), dim=1)
        return out_class, out_moe1, out_moe2, out


class PapaGEI(nn.Module):
    _CFG = dict(base_filters=32, kernel_size=3, stride=2, groups=1, n_block=18, n_classes=512, n_experts=3)

    def __init__(self, use_moe: bool = False, cfg: dict = None):
        super().__init__()
        c = cfg or self._CFG
        if use_moe:
            self.encoder = ResNet1DMoE(in_channels=1, base_filters=c["base_filters"], kernel_size=c["kernel_size"],
                                       stride=c["stride"], groups=c["groups"], n_block=c["n_block"],
                                       n_classes=c["n_classes"], n_experts=c.get("n_experts", 3))
        else:
            self.encoder = ResNet1D(in_channels=1, base_filters=c["base_filters"], kernel_size=c["kernel_size"],
                                    stride=c["stride"], groups=c["groups"], n_block=c["n_block"],
                                    n_classes=c["n_classes"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.encoder(x)[0]


def load_papagei_from_checkpoint(checkpoint_path: str, device: str = "cpu") -> PapaGEI:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    use_moe = any("expert_layers" in k for k in ckpt.keys())
    model = PapaGEI(use_moe=use_moe).to(torch.device(device))
    model.encoder = load_model_without_module_prefix(model.encoder, checkpoint_path, device=device)
    model.eval()
    print(f"Loaded PapaGEI ({'MoE' if use_moe else 'standard'}) from: {checkpoint_path}")
    return model


if __name__ == "__main__":
    import os

    ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "checkpoints", "papagei")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fname in ["papagei_s.pt", "papagei_p.pt", "papagei_s_svri.pt"]:
        ckpt_path = os.path.join(ckpt_dir, fname)
        if os.path.exists(ckpt_path):
            model = load_papagei_from_checkpoint(ckpt_path, device=device)
        else:
            print(f"Checkpoint not found: {ckpt_path}, using random weights.")
            model = PapaGEI().to(device).eval()

        x = torch.randn(2, 1250, device=device)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 512), f"Unexpected shape: {out.shape}"
        print(f"{fname}: input={x.shape} output={out.shape} OK")
