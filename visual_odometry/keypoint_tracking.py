from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy.typing import NDArray

from visual_odometry.common.enums import LogLevel
from visual_odometry.common import BaseClass
from visual_odometry.common import State
from visual_odometry.common import ParamServer


class KeypointTracker(BaseClass):
    def __init__(self, param_server: ParamServer, debug: LogLevel = LogLevel.INFO):
        """
        Key point tracker class. This class is responsible for tracking the keypoints between two frames using KLT and RANSAC.

        :param param_server: ParamServer object containing the parameters.
        :type param_server: ParamServer
        :param debug: Debug level.
        :type debug: LogLevel
        """
        super().__init__(debug)
        self._init_figure()

        self._info_print("Keypoint tracker initialized.")


        # Retrieve required parameters from the ParamServer
        self.params = param_server["keypoint_tracker"]
        
    @BaseClass.plot_debug
    def _init_figure(self):
        self.debug_fig = plt.figure() # figure for visualization
        ax = self.debug_fig.gca()

    def __call__(self, state: State, previous_image: np.ndarray, current_image: np.ndarray, K: np.ndarray) -> State:
        """Main method for keypoint tracking.
        
        :param state: State object contaiing information about the current estimate
        :type state: State
        :param previous_image: Previous image frame.
        :type previous_image: np.ndarray
        :param current_image: Current image frame.
        :type current_image: np.ndarray        
        """

        P_old = state.P.copy()
        X = state.X.copy()
        
        P_old = P_old.T.reshape(-1, 1, 2).astype(np.float32)
        P_new, status, _ = cv2.calcOpticalFlowPyrLK(prevImg = previous_image,
                                                    nextImg = current_image,
                                                    prevPts = P_old,
                                                    nextPts = None,
                                                    winSize = self.params["winSize"],
                                                    maxLevel = self.params["maxLevel"],
                                                    criteria = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.params["criteria_count"], self.params["criteria_eps"]))  


        # get the matching points
        P_new_matching = P_new[status == 1]
        P_old_matching = P_old[status == 1]

        # select corresponding 3D points (still in the world frame as was in the input)
        X_matching = X[:,status.ravel() == 1]


        # find the fundamental matrix with ransac
        #_, inliers = cv2.findFundamentalMat(P_old_matching, P_new_matching, cv2.FM_RANSAC)
        _, inliers = cv2.findEssentialMat(P_old_matching, P_new_matching, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if np.sum(inliers) < self.params["min_inliers"]:
            raise ValueError(f"Not enough inliers found. Minimum {self.params['min_inliers']} required.")
        
        P_new_matching = P_new_matching[inliers.ravel() == 1]
        P_old_matching = P_old_matching[inliers.ravel() == 1]
        X_matching = X_matching[:, inliers.ravel() == 1]
        
        state.P = P_new_matching.T # back to 2xN representation
        state.X = X_matching

        # visualization
        self._debug_visualize(image = current_image,
                             P_old_discarded = P_old[status == 0].T, # old points that are discarded
                             P_old_matching = P_old_matching.T, # old points that are matched within the new frame
                             P_new = P_new_matching.T, # new points that are matched with the old frame
                             )  
        self._info_print(f"Tracking {P_new_matching.shape[0]} keypoints, discarded by KLT: {np.sum(status == 0)}, \
                            discarded by RANSAC: {np.sum(inliers == 0)}")

        return state


    def visualize(self, *args, **kwargs):
        """Visualization method for debugging."""
        # get the axis from the figure
        ax = self.debug_fig.gca()
        ax.clear()
        ax.set_title("DEBUG VISUALIZATION - Keypoint tracking")

        # plot the image
        ax.imshow(kwargs["image"], cmap="gray")
        ax.scatter(kwargs["P_old_discarded"][0, :], kwargs["P_old_discarded"][1, :], c="r", s=2, marker="x")
        ax.scatter(kwargs["P_old_matching"][0, :], kwargs["P_old_matching"][1, :], c="b", s=2)
        ax.scatter(kwargs["P_new"][0, :], kwargs["P_new"][1, :], c="g", s=5)
        
        #for i in range(kwargs["P_old_matching"].shape[1]):  
        #    ax.plot([kwargs["P_old_matching"][0, i], kwargs["P_new"][0, i]], 
        #            [kwargs["P_old_matching"][1, i], kwargs["P_new"][1, i]], c="magenta", linewidth=1)
        
        ax.plot(
            np.stack([kwargs["P_old_matching"][0, :], kwargs["P_new"][0, :]], axis=0),
            np.stack([kwargs["P_old_matching"][1, :], kwargs["P_new"][1, :]], axis=0),
            color='magenta', linestyle='-', linewidth=1)
        
        ax.legend(["Discarded points", "Matched points", "New points"], fontsize=8)

        plt.draw()
        plt.pause(.1)
