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
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ras"
    CATEGORY = "ras"

    def apply_ras(self, model: ModelPatcher, sample_ratio: float):
        model = model.clone()
        # unpatch the model
        # this makes sure that we're wrapping the model "in a pure state"
        # the model will repatch itself later
        model.unpatch_model()
        config = RASConfig(sample_ratio=sample_ratio)
        manager = RASManager(config)
        manager.wrap_model(model)
        return (model,)


NODE_CLASS_MAPPINGS = {"RegionalAdaptiveSampling": RegionalAdaptiveSampling}
NODE_DISPLAY_NAME_MAPPING = {"RegionalAdaptiveSampling": "Regional Adaptive Sampling"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPING"]
