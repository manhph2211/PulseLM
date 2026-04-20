from .ppg_encoder.pulseppg import PulsePPG, load_pulseppg_from_checkpoint
from .ppg_encoder.papagei import PapaGEI, load_papagei_from_checkpoint

def __getattr__(name):
    if name == "MultimodalPPGLLM":
        from .pulselm import MultimodalPPGLLM
        return MultimodalPPGLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MultimodalPPGLLM",
    "PulsePPG",
    "load_pulseppg_from_checkpoint",
    "PapaGEI",
    "load_papagei_from_checkpoint",
]
