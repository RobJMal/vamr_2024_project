from dataclasses import dataclass, field
from typing import TypeVar
import numpy as np
from numpy.typing import NDArray

Pose = TypeVar('Pose', bound=NDArray)

@dataclass
class State:
    """Class for storing state information.
    num_keypoints: K for dimensionality
    """
    # Define named types inside the class
    Keypoints = np.ndarray
    Landmarks = np.ndarray

    _P: Keypoints = field(default_factory=lambda: np.empty((2, 0)))  # 2xK "valid" keypoints
    _X: Landmarks = field(default_factory=lambda: np.empty((3, 0)))  # 3xK "valid" landmarks

    @property
    def P(self) -> Keypoints:
        return self._P
    
    @P.setter
    def P(self, value: Keypoints):
        self._P = value

    @property
    def X(self) -> Landmarks:
        return self._X

    @X.setter
    def X(self, value: Landmarks):
        self._X = value
