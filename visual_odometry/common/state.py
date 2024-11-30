from dataclasses import dataclass, field
import numpy as np

@dataclass
class State:
    """Class for storing state information."""
    _P: np.ndarray = field(default_factory=lambda: np.empty((2, 0)))  # 2xK keypoints
    _X: np.ndarray = field(default_factory=lambda: np.empty((3, 0)))  # 3xK landmarks

    @property
    def P(self):
        return self._P
    
    @P.setter
    def P(self, value):
        self._P = value

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        self._X = value