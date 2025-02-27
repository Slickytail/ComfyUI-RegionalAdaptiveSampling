from .ras import RASConfig, RASManager


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

    def apply_ras(self, model, sample_ratio):
        model = model.clone()
        config = RASConfig(sample_ratio=sample_ratio)
        manager = RASManager(config)
        manager.wrap_model(model.diffusion_model)
        return (model,)


NODE_CLASS_MAPPINGS = {"RegionalAdaptiveSampling": RegionalAdaptiveSampling}
NODE_DISPLAY_NAME_MAPPING = {"RegionalAdaptiveSampling": "Regional Adaptive Sampling"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPING"]
