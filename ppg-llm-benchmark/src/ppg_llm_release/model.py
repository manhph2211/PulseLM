from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType


def get_ecg_features(ecg_encoder: nn.Module, ecg: torch.Tensor) -> torch.Tensor:
    out = ecg_encoder(ecg)
    if isinstance(out, dict):
        feats = out.get("features", None)
        if feats is None:
            feats = out.get("logits", None)
            if feats is None:
                raise ValueError("ECG encoder dict must contain 'features' or 'logits'.")
    elif isinstance(out, (tuple, list)):
        feats = out[0]
    else:
        feats = out
    return feats


def get_ppg_features(ppg_encoder: nn.Module, ppg: torch.Tensor) -> torch.Tensor:
    out = ppg_encoder(ppg)
    if isinstance(out, (tuple, list)):
        feats = out[0]
    else:
        feats = out
    return feats


class MultimodalBPQALLM(nn.Module):
    def __init__(
        self,
        llm_name: str,
        ecg_encoder: nn.Module,
        ppg_encoder: nn.Module,
        setting: str = "lora",        
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        freeze_ecg_encoder: bool = True,
        freeze_ppg_encoder: bool = True,
        ecg_feat_dim: Optional[int] = None, 
        ppg_feat_dim: Optional[int] = None,
        cache_dir: Optional[str] = None, 
        token: Optional[str] = None,
    ):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name, cache_dir=cache_dir, token=token)
        self.llm_hidden = self.llm.config.hidden_size

        # encoders
        self.ecg_encoder = ecg_encoder
        self.ppg_encoder = ppg_encoder

        # ---- LoRA / frozen / full ----
        setting = setting.lower()
        if setting == "lora":
            peft_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
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

        # ---- feature dims ----
        self.ecg_feat_dim = ecg_feat_dim
        self.ppg_feat_dim = ppg_feat_dim
        assert self.ppg_feat_dim is not None, "ppg_feat_dim must be provided to build ppg_proj in __init__"

        # ---- projector (create ONCE, in __init__) ----
        self.ppg_proj = nn.Sequential(
            nn.Linear(self.ppg_feat_dim, self.llm_hidden),
            nn.GELU(),
            nn.Linear(self.llm_hidden, self.llm_hidden),
        )

        # ---- freeze encoders ----
        if freeze_ecg_encoder:
            for p in self.ecg_encoder.parameters():
                p.requires_grad = False
            self.ecg_encoder.eval()

        if freeze_ppg_encoder:
            for p in self.ppg_encoder.parameters():
                p.requires_grad = False
            self.ppg_encoder.eval()

        # ---- ensure projector trainable ----
        for p in self.ppg_proj.parameters():
            p.requires_grad = True

        self.prefix_tokens = 2
        self.ecg_proj = None

 
    def _maybe_init_projections(
        self,
        ecg_feats: Optional[torch.Tensor] = None,
        ppg_feats: Optional[torch.Tensor] = None,
        ):
        device = next(self.llm.parameters()).device

        if (self.ppg_proj is None) and (ppg_feats is not None):
            self.ppg_feat_dim = ppg_feats.size(1)
            self.ppg_proj = nn.Linear(self.ppg_feat_dim, self.llm_hidden).to(device)

        if (self.ecg_proj is None) and (ecg_feats is not None):
            self.ecg_feat_dim = ecg_feats.size(1)
            self.ecg_proj = nn.Linear(self.ecg_feat_dim, self.llm_hidden).to(device)

 
    def _build_inputs_embeds(
            self,
            input_ids: torch.Tensor,
            ecg_token: Optional[torch.Tensor] = None,
            ppg_token: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
        embeddings = self.llm.get_input_embeddings()(input_ids)  # (B,T,H)
        prefixes = []
        if ppg_token is not None:
            prefixes.append(ppg_token)  # (B,1,H)
        if ecg_token is not None:
            prefixes.append(ecg_token)  # (B,1,H)

        if len(prefixes) == 0:
            return embeddings
        return torch.cat(prefixes + [embeddings], dim=1)  # (B, prefix+T, H)
    def _build_inputs_embeds_ppg_only(self, input_ids, ppg_tok):
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B,T,H)
        return torch.cat([ppg_tok, text_embeds], dim=1)          # (B,1+T,H)


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        ecg: Optional[torch.Tensor] = None,  # (B, 1, 5000)
        ppg: Optional[torch.Tensor] = None,  # (B, 1, 1250) or (B, 1250)
        **generate_kwargs,
        ):
        assert ppg is not None, "ppg tensor is required."

        device = next(self.llm.parameters()).device
        input_ids = input_ids.to(device)

        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        # ---- feats ----
        ecg_feats = None
        if ecg is not None:
            ecg = ecg.to(device)
            with torch.set_grad_enabled(any(p.requires_grad for p in self.ecg_encoder.parameters())):
                ecg_feats = get_ecg_features(self.ecg_encoder, ecg)  # (B, D_ecg)

        ppg = ppg.to(device)
        with torch.set_grad_enabled(any(p.requires_grad for p in self.ppg_encoder.parameters())):
            ppg_feats = get_ppg_features(self.ppg_encoder, ppg)  # (B, D_ppg)

        # ---- init proj (as needed) ----
        self._maybe_init_projections(ecg_feats=ecg_feats, ppg_feats=ppg_feats)

        # ---- build prefix tokens ----
        ppg_tok = self.ppg_proj(ppg_feats).unsqueeze(1)  # (B,1,H)

        ecg_tok = None
        if ecg_feats is not None:
            ecg_tok = self.ecg_proj(ecg_feats).unsqueeze(1)  # (B,1,H)

        inputs_embeds = self._build_inputs_embeds(input_ids, ecg_token=ecg_tok, ppg_token=ppg_tok)

        # Prefix length is 1 for PPG-only and 2 when ECG is also present.
        prefix_len = 1 + (1 if ecg_tok is not None else 0)

        if attention_mask is not None:
            prefix_pad = torch.ones(
                (attention_mask.size(0), prefix_len),
                dtype=attention_mask.dtype,
                device=device,
            )
            attention_mask = torch.cat([prefix_pad, attention_mask], dim=1)

        if labels is not None:
            ignore_pad = torch.full(
                (labels.size(0), prefix_len),
                -100,
                dtype=labels.dtype,
                device=device,
            )
            labels = torch.cat([ignore_pad, labels], dim=1)

        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return out


    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ecg: Optional[torch.Tensor] = None,
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

        # Initialize the projection layer lazily from PPG features if needed.
        self._maybe_init_projections(ecg_feats=None, ppg_feats=ppg_feats)

        ppg_tok = self.ppg_proj(ppg_feats).unsqueeze(1)  # (B,1,H)

        # Generation uses the PPG prefix only.
        inputs_embeds = self._build_inputs_embeds_ppg_only(input_ids, ppg_tok)

        # Generation always adds one prefix token for PPG.
        if attention_mask is not None:
            prefix_len = 1
            prefix_pad = torch.ones(
                (attention_mask.size(0), prefix_len),
                dtype=attention_mask.dtype,
                device=device,
            )
            attention_mask = torch.cat([prefix_pad, attention_mask], dim=1)

        gen = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        return gen
