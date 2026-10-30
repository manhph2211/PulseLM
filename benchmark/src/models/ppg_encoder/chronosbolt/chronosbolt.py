import copy
import json
import os

import torch
import torch.nn as nn
from transformers import T5Config
from transformers.activations import ACT2FN
from transformers.models.t5.modeling_t5 import T5Stack

from .._utils import load_weights, resolve_weights


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, act_fn_name, dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = ACT2FN[act_fn_name]
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        hid = self.act(self.hidden_layer(x))
        return self.dropout(self.output_layer(hid)) + self.residual_layer(x)


class ChronosBolt(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        cc = config.chronos_config
        self.patch_size = cc["input_patch_size"]
        self.patch_stride = cc["input_patch_stride"]
        self.context_length = cc["context_length"]
        self.use_reg_token = cc.get("use_reg_token", False)
        self.out_dim = config.d_model

        self.shared = nn.Embedding(2 if self.use_reg_token else 1, config.d_model)
        self.input_patch_embedding = ResidualBlock(
            self.patch_size * 2, config.d_ff, config.d_model, config.dense_act_fn, config.dropout_rate)
        enc_cfg = copy.deepcopy(config)
        enc_cfg.is_decoder = False
        enc_cfg.use_cache = False
        self.encoder = T5Stack(enc_cfg, self.shared)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        x = x[..., -self.context_length:].float()

        loc = x.mean(dim=-1, keepdim=True)
        scale = (x - loc).square().mean(dim=-1, keepdim=True).sqrt()
        scale = torch.where(scale == 0, torch.full_like(scale, 1e-5), scale)
        x = ((x - loc) / scale).to(self.shared.weight.dtype)

        mask = torch.ones_like(x)
        pad = (-x.shape[-1]) % self.patch_size
        if pad:
            x = nn.functional.pad(x, (pad, 0))
            mask = nn.functional.pad(mask, (pad, 0))
        patches = x.unfold(-1, self.patch_size, self.patch_stride)
        patch_mask = mask.unfold(-1, self.patch_size, self.patch_stride)
        embeds = self.input_patch_embedding(torch.cat([patches * patch_mask, patch_mask], dim=-1))
        attn = (patch_mask.sum(dim=-1) > 0).to(embeds.dtype)

        if self.use_reg_token:
            reg = self.shared(torch.ones((x.shape[0], 1), dtype=torch.long, device=x.device))
            embeds = torch.cat([embeds, reg], dim=1)
            attn = torch.cat([attn, torch.ones_like(attn[:, :1])], dim=1)

        hidden = self.encoder(inputs_embeds=embeds, attention_mask=attn).last_hidden_state
        n_patch = patches.shape[1]
        w = attn[:, :n_patch].unsqueeze(-1)
        return (hidden[:, :n_patch] * w).sum(dim=1) / w.sum(dim=1)


def load_chronosbolt_from_checkpoint(checkpoint_path="amazon/chronos-bolt-small", device="cpu"):
    weights = resolve_weights(checkpoint_path, "model.safetensors")
    with open(resolve_weights(checkpoint_path, "config.json") if not os.path.isfile(checkpoint_path)
              else os.path.join(os.path.dirname(weights), "config.json")) as f:
        config = T5Config(**json.load(f))
    model = ChronosBolt(config)

    state = load_weights(weights)
    keep = ("shared.", "input_patch_embedding.", "encoder.")
    state = {k: v for k, v in state.items() if k.startswith(keep)}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if k != "encoder.embed_tokens.weight"]
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)
    model.to(torch.device(device)).eval()
    print(f"Loaded Chronos-Bolt (d_model={model.out_dim}) from: {checkpoint_path}")
    return model
