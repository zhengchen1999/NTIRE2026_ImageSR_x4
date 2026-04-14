__all__ = ["MultiInputImageDataset", "FusionRestormer", "SwinIRRefiner", "build_model"]


def __getattr__(name: str):
    if name == "MultiInputImageDataset":
        from .dataset import MultiInputImageDataset

        return MultiInputImageDataset
    if name in {"FusionRestormer", "SwinIRRefiner", "build_model"}:
        from .model import FusionRestormer, SwinIRRefiner, build_model

        return {
            "FusionRestormer": FusionRestormer,
            "SwinIRRefiner": SwinIRRefiner,
            "build_model": build_model,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
