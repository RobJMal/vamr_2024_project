import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from numpy.typing import NDArray

from visual_odometry.common.enums import LogLevel
from visual_odometry.common import BaseClass
from visual_odometry.common import State
from visual_odometry.common import ParamServer
from visual_odometry.common.plot_utils import PlotUtils
from visual_odometry.common.state import Pose
from visual_odometry.pose_estimating import PoseEstimator


class Initialization(BaseClass):
    def __init__(self, param_server: ParamServer, debug: LogLevel = LogLevel.INFO):
        super().__init__(debug)
        self._init_figure()
        self._info_print("Initialization initialized.")

        # TODO: retrieve required parameters from the ParamServer


    @BaseClass.plot_debug
    def _init_figure(self):
        self.vis_figure, self.vis_axs = plt.subplots(2, 1, figsize=(10, 12))  # figure for visualization
        self.vis_axs[1].remove()
        self.vis_axs[1] = self.vis_figure.add_subplot(2, 1, 2, projection='3d')
        self.vis_axs[1].view_init(elev=-90, azim=45, roll=-45)
        self.vis_axs[1].set_xlabel("World X")
        self.vis_axs[1].set_ylabel("World Y")
        self.vis_axs[1].set_zlabel("World Z")


    def __call__(self, image_0: np.ndarray, image_1: np.ndarray, K: np.ndarray, is_KITTI: bool):
        """Main method for initialization.

        :param state: State object containing information needed for initialization
        :param image_0: First frame selected for initialization.
        :type image_0: np.ndarray
        :param image_1: Second frame selected for initialization.
        :type image_1: np.ndarray
        :return: state of last bootstrap frame with initialized inlier keypoints (state.P) and associated landmarks (state.X)
        :rtype: State
        """

        state: State = State()

        keypoints_0, keypoints_1, matches = self.get_keypoints_and_matches(image_0, image_1)
        self._debug_print(f"Before RANSAC: Number of keypoints in image_0 = {len(keypoints_0)}, Number of keypoints in image_1 = {len(keypoints_1)}, Number of matches = {len(matches)}")
        self._debug_visualize(image=cv2.drawMatches(image_0, keypoints_0, image_1, keypoints_1, matches, None), title="Keypoint matches before RANSAC")

        # Use RANSAC to estimate essential matrix, E
        points_0 = np.float32([keypoints_0[m.queryIdx].pt for m in matches])
        points_1 = np.float32([keypoints_1[m.trainIdx].pt for m in matches])
        E, mask = cv2.findEssentialMat(points_0, points_1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        # Filter inliers
        inliers_0 = points_0[mask.ravel() == 1]
        inliers_1 = points_1[mask.ravel() == 1]
        self._debug_print(f"After RANSAC: Number of inlier matches = {len(inliers_1)}")
        self._debug_visualize(image=cv2.drawMatches(image_0, keypoints_0, image_1, keypoints_1, [m for i, m in enumerate(matches) if mask[i]], None), title="Inlier matches after RANSAC")

        # Recover relative pose.
        # Frame of ref: https://stackoverflow.com/questions/77522308/understanding-cv2-recoverposes-coordinate-frame-transformations
        # The R, t rotations and translation from the first camera to the second camera.
        # Thus, 3D points represented in the frame of camera 1 can be rotated and translated using R and t to get them wrt caemra 2.
        # Notationally, R_cam2_wrt_cam1 * X_cam1 + t_cam2_wrt_cam1 = X_cam2. Thus this needs to be inverted to get the camera pose of camera 2.
        R_0_wrt_world, t_0_wrt_world = np.eye(3), np.zeros((3, 1))
        _, R_1_wrt_0, t_1_wrt_0, _ = cv2.recoverPose(E, inliers_0, inliers_1, K)

        # Triangulate points
        proj_matrix_0 = K @ np.hstack((R_0_wrt_world, t_0_wrt_world))
        proj_matrix_1 = K @ np.hstack((R_1_wrt_0, t_1_wrt_0))

        # Points are triangulated wrt image 0
        points_4D_homogenous_wrt_0 = cv2.triangulatePoints(proj_matrix_0, proj_matrix_1, inliers_0.T, inliers_1.T)

        points_3D_wrt_0 = points_4D_homogenous_wrt_0[:3, :] / points_4D_homogenous_wrt_0[3, :]
        points_3D_wrt_1 = R_1_wrt_0 @ points_3D_wrt_0 + t_1_wrt_0
        rejection_mask = (points_3D_wrt_0[2, :] < 0) | (points_3D_wrt_1[2, :] < 0)

        rejected_pts = points_3D_wrt_0[:, rejection_mask]
        self._debug_print(f"Rejecting: {rejected_pts.shape} points due to triangulation behind the camera")


        state.P = inliers_0.T[:, ~rejection_mask]
        state.X = points_3D_wrt_0[:, ~rejection_mask]
        # state.C = inliers_0.T[:, ~rejection_mask]
        # state.F = inliers_0.T[:, ~rejection_mask]

        # R_cam2_wrt_world = R # R.T
        # t_cam2_wrt_world = t # -R.T @ t
        transform = PoseEstimator.cvt_rot_trans_to_pose(np.eye(3), np.zeros((3, 1))).reshape((-1, 1)) # Convert to transform vector
        # state.Tau = np.tile(transform, state.P.shape[1])

        # Compare the bootstrapped keypoints with the keypoints from exercise 7
        if is_KITTI:
            self._debug_visualize(image=image_0, title="Initial Keypoints", points=[self.get_ex7_keypoints(), inliers_0.T])

        self.visualize_3d(state.P, state.X, transform.reshape((4, 4)))
        self._refresh_figures()

        return state
    
    @staticmethod
    def get_harris_keypoints_and_sift_descriptors(image: np.ndarray):
        """Method to get the Harris keypoints and SIFT descriptors of an image.

        :param image: Image to get the keypoints and descriptors.
        :type image: np.ndarray
        :return: Harris keypoints and SIFT descriptors.
        :rtype: [cv2.KeyPoint], np.ndarray
        """

        # Detect Shi-Tomasi corners (good features to track)
        corners = cv2.goodFeaturesToTrack(
            image,
            maxCorners=0,  # Set to 0 to detect as much as it finds
            qualityLevel=0.01,
            minDistance=10,  
            blockSize=3 
        )

        # Convert corners to cv2.KeyPoint objects
        keypoints = [cv2.KeyPoint(float(x), float(y), 1) for x, y in corners[:, 0, :]]

        # Use SIFT descriptor
        sift = cv2.SIFT_create()

        # Find descriptors
        keypoints, descriptors = sift.compute(image, keypoints)

        return keypoints, descriptors

    @staticmethod
    def get_keypoints_and_matches(image_0: np.ndarray, image_1: np.ndarray):
        """Method to get the keypoints and matches between two images.

        :param image_0: First image.
        :type image_0: np.ndarray
        :param image_1: Second image.
        :type image_1: np.ndarray
        :return: Keypoints of both images and matches.
        :rtype: [cv2.KeyPoint], [cv2.KeyPoint], [cv2.DMatch]
        """

        # Find keypoints and descriptors
        keypoints_0, descriptors_0 = Initialization.get_harris_keypoints_and_sift_descriptors(image_0)
        keypoints_1, descriptors_1 = Initialization.get_harris_keypoints_and_sift_descriptors(image_1)

        # Match descriptors
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors_0, descriptors_1)

        return keypoints_0, keypoints_1, matches

    def get_ex7_keypoints(self):
        """Method to get the keypoints from exercise 7.

        :return: Keypoints from exercise 7.
        :rtype: np.ndarray
        """

        kp = np.loadtxt(os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets/kitti/kp_for_debug.txt")).T
        kp[[0,1]] = kp[[1,0]] # swap x and y

        return kp

    @BaseClass.plot_debug
    def visualize_3d(self, keypoints, landmarks, pose: Pose):
        """
        In the 3D axis, visualize the camera pose R, t;
        visualize the keypoints, and visualize the 3D landmarks

        Convention for this task is that landmarks are wrt the world frame (camera 1)
        """
        PlotUtils._plot_pose(self.vis_axs[1], pose, isWorld=True)
        self.vis_axs[1].scatter(landmarks[0, :], landmarks[1, :], landmarks[2, :], label="Landmarks")


    def visualize(self, *args, **kwargs):
        # get the axis from the figure
        self.vis_figure.suptitle(f"DEBUG VISUALIZATION - Initialization: {kwargs['title']}")

        # plot the image
        self.vis_axs[0].imshow(kwargs['image'], cmap="gray")
        if 'points' in kwargs:
            colors = ['r', 'g']
            labels = ['Ex7 Keypoints', 'Bootstrapped Keypoints']
            for i in range(len(kwargs['points'])):
                self.vis_axs[0].scatter(kwargs['points'][i][0], kwargs['points'][i][1], c=colors[i], s=5, label=labels[i])
            self.vis_axs[0].legend()

    @BaseClass.plot_debug
    def _refresh_figures(self):
        self.vis_figure.canvas.draw_idle()
        for ax in self.vis_axs.flat:
            ax.legend()
        plt.pause(.1)

