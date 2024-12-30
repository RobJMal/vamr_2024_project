from dataclasses import dataclass, field
from typing import NewType, TypeVar
import numpy as np
from numpy.typing import NDArray

Pose = NewType('Pose', np.ndarray)

@dataclass
class State:
    """Class for storing state information.
    num_keypoints: K for dimensionality
    """
    # Define named types inside the class
    Keypoints = np.ndarray
    Landmarks = np.ndarray
    PoseVectors = np.ndarray

    _P: Keypoints = field(default_factory=lambda: np.empty((2, 0)))  # 2xK "valid" keypoints
    _X: Landmarks = field(default_factory=lambda: np.empty((3, 0)))  # 3xK "valid" landmarks
    _C: Keypoints = field(default_factory=lambda: np.empty((2, 0)))  # 2xM "candidate" keypoints
    _F: Keypoints = field(default_factory=lambda: np.empty((2, 0)))  # 2xM initial location of "candidate" keypoints
    _Tau: PoseVectors = field(default_factory=lambda: np.empty((16, 0)))  # 16xM "candidate" pose vectors always in WORLD FRAME (extrinsic)

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

    @property
    def C(self) -> Keypoints:
        return self._C

    @C.setter
    def C(self, value: Keypoints):
        self._C = value

    @property
    def F(self) -> Keypoints:
        return self._F

    @F.setter
    def F(self, value: Keypoints):
        self._F = value

    @property
    def Tau(self) -> PoseVectors:
        return self._Tau

    @Tau.setter
    def Tau(self, value: PoseVectors):
        self._Tau = value
