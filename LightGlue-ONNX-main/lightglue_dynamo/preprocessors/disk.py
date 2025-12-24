import numpy as np

from .base import PreprocessorBase


class DISKPreprocessor(PreprocessorBase):
    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        """Preprocess image from cv2.imread, (..., H, W, 3), BGR."""
        image = image / 255
        axes = [*list(range(image.ndim - 3)), -1, -3, -2]
        image = image.transpose(*axes)  # (..., 3, H, W)
        return image
