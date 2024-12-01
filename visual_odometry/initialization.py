import numpy as np
import matplotlib.pyplot as plt

from visual_odometry.common.enums import LogLevel
from visual_odometry.common import BaseClass
from visual_odometry.common import State
from visual_odometry.common import ParamServer


class Initialization(BaseClass):
    def __init__(self, param_server: ParamServer, debug: LogLevel = LogLevel.INFO):
        super().__init__(debug)
        self._info_print("Initialization initialized.")

        # TODO: retrieve required parameters from the ParamServer

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

        self.visualize(image_0, title="Image 0")
        self.visualize(image_1, title="Image 1")

        return state

    def visualize(self, image: np.ndarray, title: str = "Image"):
        """Visualization method for debugging.

        :param image: Image to be visualized.
        :type image: np.ndarray
        :param title: Title of the plot.
        :type title: str
        """
        plt.figure(self.debug_fig.number)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()