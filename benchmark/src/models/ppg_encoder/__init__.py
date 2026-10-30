import importlib

PPG_ENCODERS = {
    "papagei":     (125, 512, "papagei",     "load_papagei_from_checkpoint"),
    "pulseppg":    (50,  512, "pulseppg",    "load_pulseppg_from_checkpoint"),
    "anyppg":      (125, 512, "anyppg",      "load_anyppg_from_checkpoint"),
    "sigmappg":    (50,  360, "sigmappg",    "load_sigmappg_from_checkpoint"),
    "normwear":    (65,  768, "normwear",    "load_normwear_from_checkpoint"),
    "chronosbolt": (125, 512, "chronosbolt", "load_chronosbolt_from_checkpoint"),
}

ENCODER_HZ = {name: spec[0] for name, spec in PPG_ENCODERS.items()}


def load_ppg_encoder(name: str, checkpoint_path: str, device: str = "cpu"):
    if name not in PPG_ENCODERS:
        raise ValueError(f"Unknown PPG encoder '{name}'. Choose from: {sorted(PPG_ENCODERS)}")
    _, _, module, loader = PPG_ENCODERS[name]
    mod = importlib.import_module(f"{__name__}.{module}")
    return getattr(mod, loader)(checkpoint_path, device=device)


def encoder_feat_dim(name: str, encoder=None) -> int:
    return int(getattr(encoder, "out_dim", None) or PPG_ENCODERS[name][1])
