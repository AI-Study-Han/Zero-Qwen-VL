
import warnings

from transformers.utils import logging
from .image_processing_qwenvl import QWENVLImageProcessor


logger = logging.get_logger(__name__)


class QWENVLFeatureExtractor(QWENVLImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "The class CLIPFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use CLIPImageProcessor instead.",
            FutureWarning,
        )
        super().__init__(*args, **kwargs)
