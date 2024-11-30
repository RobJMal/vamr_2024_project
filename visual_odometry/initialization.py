import numpy as np
import matplotlib.pyplot as plt

from visual_odometry.common.enums import LogLevel
from visual_odometry.common import BaseClass
from visual_odometry.common import State


class Initialization(BaseClass):
    def __init__(self, params: dict, debug: LogLevel = LogLevel.TEXT):
        super().__init__(debug)
        self._debug_print("Initialization initialized.")

        self.params = params  # Parameters for initialization
        self.debug_fig = plt.figure()  # figure for visualization

    def __call__(self, image_0: np.ndarray, image_1: np.ndarray):
        """Main method for initialization.

        :param state: State object containing information needed for initialization
        :param image_0: First frame selected for initialization.
        :type image_0: np.ndarray
        :param image_1: Second frame selected for initialization.
        :type image_1: np.ndarray        
        """
        # returns state object with initialized inlier keypoints (state.P) and associated landmarks (state.X)

        state: State = State()

        pass

    def visualize(self, *args, **kwargs):
        """Visualization method for debugging."""
