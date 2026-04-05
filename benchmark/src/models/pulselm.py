from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType


def get_ppg_features(ppg_encoder: nn.Module, ppg: torch.Tensor) -> torch.Tensor:
    out = ppg_encoder(ppg)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


class MultimodalPPGLLM(nn.Module):
    def __init__(
        self,
        llm_name: str,
        ppg_encoder: nn.Module,
        ppg_feat_dim: int,
        setting: str = "lora",
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        freeze_ppg_encoder: bool = True,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
    ):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name, cache_dir=cache_dir, token=token)
        self.llm_hidden = self.llm.config.hidden_size
        self.ppg_encoder = ppg_encoder

        setting = setting.lower()
        if setting == "lora":
            peft_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            self.llm = get_peft_model(self.llm, peft_cfg)
        elif setting == "frozen":
            for p in self.llm.parameters():
                p.requires_grad = False
            self.llm.eval()
        elif setting == "full":
            pass
        else:
            raise ValueError("setting must be one of: 'lora' | 'frozen' | 'full'")

        self.ppg_feat_dim = ppg_feat_dim
        self.ppg_proj = nn.Sequential(
            nn.Linear(self.ppg_feat_dim, self.llm_hidden),
            nn.GELU(),
            nn.Linear(self.llm_hidden, self.llm_hidden),
        )

        if freeze_ppg_encoder:
            for p in self.ppg_encoder.parameters():
                p.requires_grad = False
            self.ppg_encoder.eval()

        for p in self.ppg_proj.parameters():
            p.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        ppg: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        assert ppg is not None, "ppg tensor is required."
        device = next(self.llm.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        ppg = ppg.to(device)
        with torch.set_grad_enabled(any(p.requires_grad for p in self.ppg_encoder.parameters())):
            ppg_feats = get_ppg_features(self.ppg_encoder, ppg)

        ppg_tok = self.ppg_proj(ppg_feats).unsqueeze(1)
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([ppg_tok, text_embeds], dim=1)

        if attention_mask is not None:
            attention_mask = torch.cat(
                [torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=device), attention_mask],
                dim=1)

        if labels is not None:
            labels = torch.cat(
                [torch.full((labels.size(0), 1), -100, dtype=labels.dtype, device=device), labels],
                dim=1)

        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ppg: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        **kwargs,
    ):
        assert ppg is not None, "ppg tensor is required."
        device = next(self.llm.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        ppg = ppg.to(device)
        ppg_feats = get_ppg_features(self.ppg_encoder, ppg)
        ppg_tok = self.ppg_proj(ppg_feats).unsqueeze(1)
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([ppg_tok, text_embeds], dim=1)

        if attention_mask is not None:
            attention_mask = torch.cat(
                [torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=device), attention_mask],
                dim=1)

        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
