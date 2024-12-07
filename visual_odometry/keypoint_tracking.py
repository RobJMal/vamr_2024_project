import numpy as np
import matplotlib.pyplot as plt
import cv2

from visual_odometry.common.enums import LogLevel
from visual_odometry.common import BaseClass
from visual_odometry.common import State
from visual_odometry.common import ParamServer


class KeypointTracker(BaseClass):
    def __init__(self, param_server: ParamServer, debug: LogLevel = LogLevel.INFO):
        super().__init__(debug)
        self._info_print("Keypoint tracker initialized.")

        # TODO: retrieve required parameters from the ParamServer
        self.params = param_server["keypoint_tracker"]



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
        # TODO: might be enough to pass P and return the new P -> but then there is no visualization
        # specifically this method only needs state.P to return a new state P and previous_image, current_image

        P_old = state.P.copy()

        P_old = P_old.T.reshape(-1, 1, 2).astype(np.float32)
        
        P_new, status, _ = cv2.calcOpticalFlowPyrLK(prevImg = previous_image,
                                                    nextImg = current_image,
                                                    prevPts = P_old,
                                                    nextPts = None,
                                                    winSize = self.params["winSize"],
                                                    maxLevel = self.params["maxLevel"],
                                                    criteria = self.params["criteria"],)

        # get the matching points
        P_new = P_new[status == 1]
        P_old = P_old[status == 1]


        # TODO: ourlier rejection ?
        


        # visualization
        self._debug_visuaize(image = current_image,
                             P_old = P_old,
                             P_new = P_new)
        self._debug_print(f"Keypoint tracking: {len(P_new)} keypoints tracked.")


    def visualize(self, *args, **kwargs):
        """Visualization method for debugging."""
        # get the axis from the figure
        ax = self.debug_fig.gca()
        ax.clear()

        # plot the image
        ax.imshow(self.current_image)


        
    
