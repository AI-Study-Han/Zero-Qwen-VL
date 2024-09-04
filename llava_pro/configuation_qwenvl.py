import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union


if TYPE_CHECKING:
    from transformers.processing_utils import ProcessorMixin
    from transformers.utils import TensorType

from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import OnnxConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class QWENVisionConfig(PretrainedConfig):
    model_type = "qwen_vision_model"

    def __init__(
        self,
        heads=16,
        image_size=448,
        image_start_id=0000,
        layers=48,
        mlp_ratio=4.9231,
        output_dim=4096,
        patch_size=14,
        width=1664,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.heads = heads
        self.image_size = image_size
        self.image_start_id = image_start_id
        self.layers = layers
        self.mlp_ratio = mlp_ratio
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.width = width

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from CLIPConfig
        if config_dict.get("model_type") == "clip":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)