

import warnings

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import CONFIG_MAPPING
from llava_pro.configuation_qwenvl import QWENVisionConfig

logger = logging.get_logger(__name__)


class LlavaproConfig(PretrainedConfig):
   

    model_type = "llavapro"
    is_composition = False

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=151646,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act

        if vision_feature_select_strategy not in ["default", "full", "pooler"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full', 'pooler',."
                f"Got: {vision_feature_select_strategy}"
            )

        if "vocab_size" in kwargs:
            warnings.warn(
                "The `vocab_size` argument is deprecated and will be removed in v4.42, since it can be inferred from the `text_config`. Passing this argument has no effect",
                FutureWarning,
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer

        if isinstance(vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "my_custom_vision_model"
            )
            vision_config = QWENVisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = QWENVisionConfig(
                heads=16,
                image_size=448,
                image_start_id=0000,
                layers=48,
                mlp_ratio=4.9231,
                output_dim=4096,
                patch_size=14,
                width=1664,
            )
        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()
            
        self.text_config = text_config

        super().__init__(**kwargs)
