from .ras import RASConfig, RASManager
from comfy.model_patcher import ModelPatcher


class RegionalAdaptiveSampling:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "sample_ratio": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.05},
                ),
                "warmup_steps": (
                    "INT",
                    {"default": 4, "min": 0, "max": 100},
                ),
                "hydrate_every": (
                    "INT",
                    {"default": 4, "min": 0, "max": 100},
                ),
                "starvation_scale": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ras"
    CATEGORY = "ras"

    def apply_ras(
        self,
        model: ModelPatcher,
        sample_ratio: float,
        warmup_steps: int,
        hydrate_every: int,
        starvation_scale: float,
    ):
        model = model.clone()
        # unpatch the model
        # this makes sure that we're wrapping the model "in a pure state"
        # the model will repatch itself later
        model.unpatch_model()
        config = RASConfig(
            sample_ratio=sample_ratio,
            warmup_steps=warmup_steps,
            hydrate_every=hydrate_every,
            starvation_scale=starvation_scale,
        )
        manager = RASManager(config)
        manager.wrap_model(model)
        return (model,)


NODE_CLASS_MAPPINGS = {"RegionalAdaptiveSampling": RegionalAdaptiveSampling}
NODE_DISPLAY_NAME_MAPPING = {"RegionalAdaptiveSampling": "Regional Adaptive Sampling"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPING"]
