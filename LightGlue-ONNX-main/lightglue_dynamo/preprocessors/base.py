from abc import ABC, abstractmethod

import numpy as np


class PreprocessorBase(ABC):
    @staticmethod
    @abstractmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        pass
