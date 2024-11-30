import numpy as np
import matplotlib.pyplot as plt

from visual_odometry.common.enums import LogLevel
from visual_odometry.common import BaseClass
from visual_odometry.common import State


class KeypointTracker(BaseClass):
    def __init__(self, params: dict, debug: LogLevel = LogLevel.TEXT):
        super().__init__(debug)
        self._debug_print("Keypoint tracker initialized.")

        self.params = params # Parameters for keypoint tracking
        # INFO: required params for tracking: tracker params,
        self.debug_fig = plt.figure() # figure for visualization

    def __call__(self, state: State, previous_image: np.ndarray, current_image: np.ndarray):
        """Main method for keypoint tracking.
        
        :param state: State object contaiing information about the current estimate
        :type keypoints: np.ndarray
        :param previous_image: Previous image frame.
        :type previous_image: np.ndarray
        :param current_image: Current image frame.
        :type current_image: np.ndarray        
        """
        # specifically this method only needs state.P!

        pass


    def visualize(self, *args, **kwargs):
        """Visualization method for debugging."""

        
    
