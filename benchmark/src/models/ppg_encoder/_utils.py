import os

import torch


def zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + eps)


def minmax(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    lo = x.amin(dim=-1, keepdim=True)
    hi = x.amax(dim=-1, keepdim=True)
    return (x - lo) / (hi - lo + eps)


def resolve_weights(path_or_repo: str, filename: str = "model.safetensors") -> str:
    if os.path.isfile(path_or_repo):
        return path_or_repo
    if os.path.isdir(path_or_repo):
        local = os.path.join(path_or_repo, filename)
        if not os.path.isfile(local):
            raise FileNotFoundError(f"{filename} not found in {path_or_repo}")
        return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download(path_or_repo, filename, token=os.environ.get("HF_TOKEN"))


def load_weights(path: str) -> dict:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu")
