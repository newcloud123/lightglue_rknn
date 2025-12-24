import numpy as np

from .base import PreprocessorBase


class SuperPointPreprocessor(PreprocessorBase):
    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        """Preprocess image from cv2.imread, (..., H, W, 3), BGR."""
        image = image[..., ::-1] / 255 * [0.299, 0.587, 0.114]
        image = image.sum(axis=-1, keepdims=True)  # (..., H, W, 1)
        axes = [*list(range(image.ndim - 3)), -1, -3, -2]
        image = image.transpose(*axes)  # (..., 1, H, W)
        return image
