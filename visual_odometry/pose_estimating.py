import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy.typing import NDArray

from visual_odometry.common.enums import LogLevel
from visual_odometry.common import BaseClass
from visual_odometry.common import State
from visual_odometry.common import ParamServer
from visual_odometry.common.state import Pose

class PoseEstimator(BaseClass):
    def __init__(self, param_server: ParamServer, debug: LogLevel = LogLevel.INFO):
        """
        Pose estimator class. This class is responsible for estimating the camera pose from the tracked keypoints.
        
        :param param_server: ParamServer object containing the parameters.
        :type param_server: ParamServer
        :param debug: Debug level.
        :type debug: LogLevel
        """
        super().__init__(debug)
        self._info_print("Pose estimator initialized.")
        
        # Retrieve required parameters from the ParamServer
        self.params = param_server["pose_estimator"]
        
        self.debug_fig = plt.figure()

    @staticmethod
    def cvt_rot_trans_to_pose(rot_matrix: NDArray, trans_vec: NDArray) -> NDArray:
        return np.block([
            [rot_matrix, trans_vec],
            [np.zeros((1, 3)), 1]
            ])


    def __call__(self, state: State, K_matrix: np.ndarray):
        """
        Main method for pose estimation.

        :param state: State object containing information about keypoints (2 x K) and their corresponding landmarks (3 x K)
        :type State: State 
        :param K:
        :type: K: np.ndarray
        """
        # Assuming no distortion
        distortion_matrix = np.zeros((1,5))
        
        num_landmarks = state.X.shape[1]
        num_keypoints = state.P.shape[1]
        min_point_correspondence = min(num_keypoints, num_landmarks)
    
        # Transposing since cv2 inverts rows and cols representation
        X_clipped = state.X[:, :min_point_correspondence].T
        P_clipped = state.P[:, :min_point_correspondence].T

        ret_val, rot_vec, trans_vec, inliers = cv2.solvePnPRansac(X_clipped, P_clipped, K_matrix, 
                                                                  distCoeffs=distortion_matrix, 
                                                                  confidence=0.999)
        rot_matrix, _ = cv2.Rodrigues(rot_vec)

        if ret_val:
            self._debug_print(f"Rotation Matrix: {rot_matrix}")
            self._debug_print(f"Translation Vector: {trans_vec}")
            self._debug_print(f"Inliers: {inliers}")
        else:
            self._debug_print("Pose estimation failed.")

        return rot_matrix, trans_vec
