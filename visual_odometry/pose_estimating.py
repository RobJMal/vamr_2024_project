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
from visual_odometry.common.utils import Utils

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


    def __call__(self, state: State, K_matrix: np.ndarray, init_pose: Pose, frame_id: int=0) -> Tuple[bool, NDArray, NDArray]:
        """
        Main method for pose estimation.

        Return the R, t such that we are able to convert a world point X_w into the camera frame T_{CW}

        :param state: State object containing information about keypoints (2 x K) and their corresponding landmarks (3 x K)
        :type State: State
        :param K:
        :type: K: np.ndarray
        :param init_pose: Initial pose of the camera.
        :type init_pose: Pose
        """
        # Assuming no distortion
        distortion_matrix = np.zeros((1,5))

        # Transposing since cv2 inverts rows and cols representation
        ret_val, rot_vec_cam_wrt_w, trans_vec_cam_wrt_w, inliers = cv2.solvePnPRansac(
            objectPoints=state.X.T,
            imagePoints=state.P.T,
            cameraMatrix=K_matrix,
            distCoeffs=distortion_matrix,
            iterationsCount=self.params["pnp_ransac_iterations"],
            reprojectionError=self.params["pnp_ransac_reprojection_error"],
            confidence=self.params["pnp_ransac_confidence"]
        )

        # self._plot_pose_and_landmarks((0, 0), init_pose, state, plot_title="Pose and Landmarks")

        # Applying nonlinear optimization using inliers
        if self.params["use_reprojection_error_optimization"]:
            # Extracting inliers
            landmarks_inliers = state.X[:, inliers.flatten()]
            keypoints_inliers = state.P[:, inliers.flatten()]
            state_inliers = State(keypoints_inliers, landmarks_inliers)

            rot_vec_cam_wrt_w, trans_vec_cam_wrt_w = cv2.solvePnPRefineLM(
                objectPoints=landmarks_inliers.T,
                imagePoints=keypoints_inliers.T,
                cameraMatrix=K_matrix,
                distCoeffs=distortion_matrix,
                rvec=rot_vec_cam_wrt_w,
                tvec=trans_vec_cam_wrt_w
            )

            # Visualizing the inliers used for pose estimation after optimization
            if self.debug >= LogLevel.VISUALIZATION:
                rot_matrix_wrt_camera_vis, _ = cv2.Rodrigues(rot_vec_cam_wrt_w)

                pose_estimation_with_inliers = self.cvt_rot_trans_to_pose(rot_matrix_wrt_camera_vis, trans_vec_cam_wrt_w)

                self._plot_pose_and_landmarks((0, 1), pose_estimation_with_inliers, state_inliers, plot_title="Pose and Landmarks (using Inliers)")
                self._plot_inliers_percentage_history((1, 0), state, state_inliers, frame_id=frame_id)
                self._plot_num_inliers_history((1, 1), state, frame_id=frame_id)

        # Converting output to world frame
        rot_matrix_cam_wrt_w, _ = cv2.Rodrigues(rot_vec_cam_wrt_w)

        if ret_val:
            self._debug_print(f"Rotation Matrix (wrt world frame): {rot_matrix_cam_wrt_w}")
            self._debug_print(f"Translation Vector (wrt to world frame): {trans_vec_cam_wrt_w}")
        else:
            self._info_print("Pose estimation failed.")

        return ret_val, rot_matrix_cam_wrt_w, trans_vec_cam_wrt_w

    # region Visualization Debugging
    @BaseClass.plot_debug
    def _init_figure(self):
        self.vis_figure, self.vis_axs = plt.subplots(2, 2, figsize=(20, 8))
        self.vis_figure.suptitle("DEBUG VISUALIZATION: Pose Estimation")

    @BaseClass.plot_debug
    def _plot_pose_and_landmarks(self, fig_id: Tuple[int, int], pose: Pose, state: State, plot_title: str = "Pose and Landmarks"):
        """
        Plots the camera pose and the landmarks in the world frame.

        :param pose: Camera pose in the world frame.
        :type pose: Pose
        :param state: State object containing the landmarks.
        :type state: State
        """
        self.vis_axs[*fig_id].clear()

        self.vis_axs[*fig_id].set_title(plot_title)
        PlotUtils._plot_trajectory(self.vis_axs[*fig_id], pose, frame_id=0, plot_ground_truth=False)
        PlotUtils._plot_landmarks(self.vis_axs[*fig_id], pose, state, frame_id=0)
        self.vis_axs[*fig_id].legend()
        self.vis_axs[*fig_id].set_xlabel("X")
        self.vis_axs[*fig_id].set_ylabel("Z")

    @BaseClass.plot_debug
    def _plot_inliers_percentage_history(self, fig_id: Tuple[int, int], state: State, state_inliers: State, frame_id: int = 0):
        """
        Plots the number of inliers over time.
        """
        # Maintain history of frames and keypoint counts. This is
        # enable us to plot the history of keypoints tracked as a line
        if not hasattr(self, 'inlier_percentage_history'):
            self.inlier_percentage_history = {'frames': [], 'total_state': [], 'inliers': [], 'inliers_percentage': []}

        # Append current frame and keypoint count to the history.
        self.inlier_percentage_history['frames'].append(frame_id)
        self.inlier_percentage_history['total_state'].append(state.P.shape[1])
        self.inlier_percentage_history['inliers'].append(state_inliers.P.shape[1])
        self.inlier_percentage_history['inliers_percentage'].append(state_inliers.P.shape[1] / state.P.shape[1])

        # Clear the axis for fresh plotting
        self.vis_axs[*fig_id].clear()

        self.vis_axs[*fig_id].set_title("Inlier Tracking Count")
        self.vis_axs[*fig_id].plot(self.inlier_percentage_history['frames'], self.inlier_percentage_history['inliers_percentage'], marker='o')

        self.vis_axs[*fig_id].set_xlabel("Frame")
        self.vis_axs[*fig_id].set_ylabel("Percentage of inliers")

    @BaseClass.plot_debug
    def _plot_num_inliers_history(self, fig_id: Tuple[int, int], state: State, frame_id: int = 0):
        """
        Plots the number of inliers over time.
        """
        # Maintain history of frames and keypoint counts. This is
        # enable us to plot the history of keypoints tracked as a line
        if not hasattr(self, 'inlier_history'):
            self.inlier_history = {'frames': [], 'inliers': []}

        # Append current frame and keypoint count to the history.
        self.inlier_history['frames'].append(frame_id)
        self.inlier_history['inliers'].append(state.P.shape[1])

        # Clear the axis for fresh plotting
        self.vis_axs[*fig_id].clear()

        self.vis_axs[*fig_id].set_title("Inlier Tracking Count")
        self.vis_axs[*fig_id].plot(self.inlier_history['frames'], self.inlier_history['inliers'], marker='o')
        self.vis_axs[*fig_id].set_xlabel("Frame")
        self.vis_axs[*fig_id].set_ylabel("Number of inliers")
    # endregion
