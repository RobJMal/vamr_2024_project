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
        
    @BaseClass.plot_debug
    def _init_figure(self):
        self.debug_fig = plt.figure()

    @staticmethod
    def cvt_rot_trans_to_pose(rot_matrix: NDArray, trans_vec: NDArray) -> Pose:
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

        # Transposing since cv2 inverts rows and cols representation
        ret_val, rot_vec_wrt_camera, trans_vec_wrt_camera, inliers = cv2.solvePnPRansac(state.X.T, state.P.T, K_matrix, 
                                                                  useExtrinsicGuess=False,
                                                                  distCoeffs=distortion_matrix, 
                                                                  confidence=0.999)
        rot_matrix_wrt_camera, _ = cv2.Rodrigues(rot_vec_wrt_camera)

        rot_matrix_wrt_world = rot_matrix_wrt_camera.T
        trans_vec_wrt_world = -rot_matrix_wrt_world @ trans_vec_wrt_camera

        if ret_val:
            self._debug_print(f"Rotation Matrix (wrt world frame): {rot_matrix_wrt_world}")
            self._debug_print(f"Translation Vector (wrt to world frame): {trans_vec_wrt_world}")
            self._debug_print(f"Inliers: {inliers}")
        else:
            self._debug_print("Pose estimation failed.")

        return rot_matrix_wrt_world, trans_vec_wrt_world
