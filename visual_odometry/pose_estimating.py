from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy.typing import NDArray

from visual_odometry.common.enums import LogLevel
from visual_odometry.common import BaseClass
from visual_odometry.common import State
from visual_odometry.common import ParamServer
from visual_odometry.common.plot_utils import PlotUtils
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
        self._init_figure()
        self._info_print("Pose estimator initialized.")
        
        # Retrieve required parameters from the ParamServer
        self.params = param_server["pose_estimator"]
        
    @staticmethod
    def cvt_rot_trans_to_pose(rot_matrix: NDArray, trans_vec: NDArray) -> Pose:
        """
        Converts rotation matrix and translation vector into a pose matrix.
        """
        if trans_vec.shape != (3, 1):
            trans_vec = trans_vec.reshape(3, 1)

        return np.block([
            [rot_matrix, trans_vec],
            [np.zeros((1, 3)), 1]
            ])


    def __call__(self, state: State, K_matrix: np.ndarray) -> Tuple[bool, NDArray, NDArray]:
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
                                                                  distCoeffs=distortion_matrix, 
                                                                  useExtrinsicGuess=self.params["use_extrinsic_guess"],
                                                                  iterationsCount=self.params["pnp_ransac_iterations"],
                                                                  reprojectionError=self.params["pnp_ransac_reprojection_error"],
                                                                  confidence=self.params["pnp_ransac_confidence"])

        landmarks_inliers = state.X[:, inliers.flatten()]
        keypoints_inliers = state.P[:, inliers.flatten()]
        state_inliers = State(keypoints_inliers, landmarks_inliers)

        # Applying nonlinear optimization using inliers 
        if self.params["use_reprojection_error_optimization"]:
            rot_vec_wrt_camera, trans_vec_wrt_camera = cv2.solvePnPRefineLM(landmarks_inliers.T, keypoints_inliers.T, K_matrix.T, 
                                                                               distCoeffs=distortion_matrix,
                                                                               rvec=rot_vec_wrt_camera, tvec=trans_vec_wrt_camera)
            
            if self.debug >= LogLevel.VISUALIZATION:
                rot_matrix_wrt_camera_vis, _ = cv2.Rodrigues(rot_vec_wrt_camera)

                # Applying transform to make it wrt world frame
                rot_matrix_wrt_world_vis = rot_matrix_wrt_camera_vis.T
                trans_vector_wrt_world_vis = -rot_matrix_wrt_world_vis @ trans_vec_wrt_camera

                pose_estimation_with_inliers = self.cvt_rot_trans_to_pose(rot_matrix_wrt_world_vis, trans_vector_wrt_world_vis)
                self._plot_pose_and_landmarks((0, 0), pose_estimation_with_inliers, state_inliers)

        rot_matrix_wrt_camera, _ = cv2.Rodrigues(rot_vec_wrt_camera)

        rot_matrix_wrt_world = rot_matrix_wrt_camera.T
        trans_vec_wrt_world = -rot_matrix_wrt_world @ trans_vec_wrt_camera

        success = True
        if ret_val:
            self._debug_print(f"Rotation Matrix (wrt world frame): {rot_matrix_wrt_world}")
            self._debug_print(f"Translation Vector (wrt to world frame): {trans_vec_wrt_world}")
        else:
            success = False
            self._info_print("Pose estimation failed.")

        return success, rot_matrix_wrt_world, trans_vec_wrt_world

    # region Visualization Debugging
    @BaseClass.plot_debug
    def _init_figure(self):
        self.vis_figure, self.vis_axs = plt.subplots(2, 2, figsize=(20, 8))
        self.vis_figure.suptitle("DEBUG VISUALIZATION: Pose Estimation")

    @BaseClass.plot_debug
    def _plot_pose_and_landmarks(self, fig_id: Tuple[int, int], pose: Pose, state: State):
        """
        Plots the camera pose and the landmarks in the world frame.

        :param pose: Camera pose in the world frame.
        :type pose: Pose
        :param state: State object containing the landmarks.
        :type state: State
        """
        self.vis_axs[*fig_id].clear()

        self.vis_axs[*fig_id].set_title("Pose and Landmarks (using inliers only)")
        PlotUtils._plot_trajectory(self.vis_axs[*fig_id], pose, frame_id=0, plot_ground_truth=False)
        PlotUtils._plot_landmarks(self.vis_axs[*fig_id], pose, state, frame_id=0)
        self.vis_axs[*fig_id].legend()
        self.vis_axs[*fig_id].set_xlabel("X")
        self.vis_axs[*fig_id].set_ylabel("Z")

    # endregion