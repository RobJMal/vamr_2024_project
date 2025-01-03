from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray

from visual_odometry.common.state import Pose, State


class PlotUtils:
    """
    Static methods for plotting given a plt axs object
    """

    @staticmethod
    def _plot_pose(axs: Axes, pose: Pose, isWorld: bool):
        """
        Pose is always extrinsic (camera frame wrt world frame) so we can directly plot is.
        """
        # Camera position (origin of the camera frame)
        scale = 1
        R = pose[:3, :3]
        t = pose[:3, 3]

        x_axis = (R[:, 0] - t) * scale
        y_axis = (R[:, 1] - t) * scale
        z_axis = (R[:, 2] - t) * scale

        # Plot the camera position as a red dot
        axs.scatter(t[0], t[1], t[2], color='black' if isWorld else 'brown', s=10)

        # Plot the camera axes (X, Y, Z axes) using the rotation matrix
        axs.quiver(t[0], t[1], t[2], x_axis[0], x_axis[1], x_axis[2], color='r', length=scale)
        axs.quiver(t[0], t[1], t[2], y_axis[0], y_axis[1], y_axis[2], color='g', length=scale)
        axs.quiver(t[0], t[1], t[2], z_axis[0], z_axis[1], z_axis[2], color='b', length=scale)

    @staticmethod
    def _plot_trajectory_and_landmarks(axs: Axes, pose: Pose, state: State):
        """
        Plots the trajectory and the landmarks. Plots only the x and z coordinates since the camera
        is moving on a flat plane.

        """
        # Camera pose and landmarks wrt world frame
        camera_t_wrt_world = pose[:3, 3]
        landmarks_wrt_world = state.X

        axs.scatter(camera_t_wrt_world[0], camera_t_wrt_world[2], color='red', s=10)
        axs.scatter(landmarks_wrt_world[0, :], landmarks_wrt_world[2, :], color='black', s=10)
        axs.set_xlabel("X position")
        axs.set_ylabel("Z position")

    @staticmethod
    def _convert_pixels_to_world(keypoints: State.Keypoints, pose: Pose) -> NDArray:
        """
        keypoints are represented in the camera frame in pixels.
        Using the K (Intrinsic Parameter Matrix) of the camera, we can get the field of view to understand how wide the camera can view and accordingly
        scale the keypoints from pixel to meter scale

        @param landmarks: NOTE THIS IS IN THE CAMERA FRAME

        Idea:
            From the intrinsic matrix, we find out the field of view of the camera (horizontal only matters for plotting since y is considered to be constant).
            For a given keypoint, there's an associated landmark. If we know the z axis of the landmark, we can then scale to find out what the width of the keypoint is
        """
        average_car_width = 2 # meters
        kp_X_scaled = (keypoints[0, :] - np.min(keypoints[0, :]))/(np.max(keypoints[0, :]) - np.min(keypoints[0, :])) * average_car_width
        kp_X_scaled = kp_X_scaled + pose[0, 3] - (average_car_width/2) # Centers it to the current pose
        return np.vstack((kp_X_scaled, np.zeros_like(kp_X_scaled)))


    @staticmethod
    def _plot_keypoints_and_landmarks(axs: Axes, pose: Pose, keypoints_scaled: State.Keypoints, landmarks_wrt_camera: State.Landmarks, candidates=False):
        """
        Plots the keypoints and the landmarks.

        For keypoints: Plots on the x coordinates since it's 2D projected onto 1D
        For landmarks: Plots only the x and z coordinates since the camera is moving on a flat plane.
        """
        kp_color = "purple" if candidates else "blue"
        landmark_color = "orange" if candidates else "green"

        axs.scatter(pose[0, 3], 0, color='red', s=20, label="Curr X")
        axs.scatter(keypoints_scaled[0, :], np.zeros_like(keypoints_scaled[0, :]), color=kp_color, s=10, label=f"Keypoints {'candidates' if candidates else ''}", alpha=0.3)
        axs.scatter(landmarks_wrt_camera[0, :], landmarks_wrt_camera[2, :], color=landmark_color, s=10, label=f"Landmarks {'candidates' if candidates else ''}", alpha=0.3)

        axs.set_xlabel("X position")
        axs.set_ylabel("Z position")
