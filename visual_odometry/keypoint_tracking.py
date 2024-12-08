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
                                                    criteria = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.params["criteria_count"], self.params["criteria_eps"]))  


        # get the matching points
        P_new_matching = P_new[status == 1]
        P_old_matching = P_old[status == 1]

        # find the fundamental matrix with ransac
        _, inliers = cv2.findFundamentalMat(P_old_matching, P_new_matching, cv2.FM_RANSAC)
        if np.sum(inliers) < self.params["min_inliers"]:
            raise ValueError(f"Not enough inliers found. Minimum {self.params['min_inliers']} required.")
        
        P_new_matching = P_new_matching[inliers.ravel() == 1]
        P_old_matching = P_old_matching[inliers.ravel() == 1]
        
        state.P = P_new_matching.T

        # visualization
        self._debug_visuaize(image = current_image,
                             P_old_discarded = P_old[status == 0].T, # old points that are discarded
                             P_old_matching = P_old_matching.T, # old points that are matched within the new frame
                             P_new = P_new_matching.T, # new points that are matched with the old frame
                             )  
        self._debug_print(f"Keypoint tracking: {len(P_new)} keypoints tracked.")

        return state


    def visualize(self, *args, **kwargs):
        """Visualization method for debugging."""
        # get the axis from the figure
        ax = self.debug_fig.gca()
        ax.clear()

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

        plt.draw()
        plt.pause(.1)