import numpy as np
import matplotlib.pyplot as plt

from visual_odometry.common.enums import LogLevel
from visual_odometry.common import BaseClass
from visual_odometry.common import State
from visual_odometry.common import ParamServer


class KeypointTracker(BaseClass):
    def __init__(self, param_server: ParamServer, debug: LogLevel = LogLevel.INFO):
        super().__init__(debug)
        self._info_print("Keypoint tracker initialized.")

        # TODO: retrieve required parameters from the ParamServer



        # INFO: required params for tracking: tracker params,
        self.debug_fig = plt.figure() # figure for visualization

    def __call__(self, state: State, previous_image: np.ndarray, current_image: np.ndarray):
        """Main method for keypoint tracking.
        
        :param state: State object contaiing information about the current estimate
        :type state: State
        :param previous_image: Previous image frame.
        :type previous_image: np.ndarray
        :param current_image: Current image frame.
        :type current_image: np.ndarray        
        """
        # specifically this method only needs state.P to return a new state P and previous_image, current_image

        pass


    def visualize(self, *args, **kwargs):
        """Visualization method for debugging."""
        pass

        
    
