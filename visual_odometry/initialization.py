import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from visual_odometry.common.enums import LogLevel
from visual_odometry.common import BaseClass
from visual_odometry.common import State
from visual_odometry.common import ParamServer
from visual_odometry.pose_estimating import PoseEstimator


class Initialization(BaseClass):
    def __init__(self, param_server: ParamServer, debug: LogLevel = LogLevel.INFO):
        super().__init__(debug)
        self._info_print("Initialization initialized.")

        # TODO: retrieve required parameters from the ParamServer


    @BaseClass.plot_debug
    def _init_figure(self):
        self.debug_fig = plt.figure()  # figure for visualization
        self.ax = self.debug_fig.gca()


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
        _, R, t, _ = cv2.recoverPose(E, inliers_0, inliers_1, K)

        # Triangulate points
        proj_matrix_0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        proj_matrix_1 = K @ np.hstack((R, t))
        points_4D_homogenous = cv2.triangulatePoints(proj_matrix_0, proj_matrix_1, inliers_0.T, inliers_1.T)
        points_3D = points_4D_homogenous[:3, :] / points_4D_homogenous[3, :]

        state.P = inliers_1.T
        state.X = points_3D
        state.C = inliers_1.T
        state.F = inliers_1.T

        R_cam2_wrt_world = R.T
        t_cam2_wrt_world = -R.T @ t
        transform = PoseEstimator.cvt_rot_trans_to_pose(R_cam2_wrt_world, t_cam2_wrt_world).reshape((-1, 1)) # Convert to transform vector
        state.Tau = np.tile(transform, inliers_1.shape[0])

        # Compare the bootstrapped keypoints with the keypoints from exercise 7
        if is_KITTI:
            self._debug_visualize(image=image_0, title="Initial Keypoints", points=[self.get_ex7_keypoints(), inliers_0.T])

        return state

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

        # Use SIFT descriptor
        sift = cv2.SIFT_create()

        # Find keypoints and descriptors
        keypoints_0, descriptors_0 = sift.detectAndCompute(image_0, None)
        keypoints_1, descriptors_1 = sift.detectAndCompute(image_1, None)

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

    def visualize(self, *args, **kwargs):
        # get the axis from the figure
        plt.title(f"DEBUG VISUALIZATION - Initialization: {kwargs['title']}")

        # plot the image
        plt.imshow(kwargs['image'], cmap="gray")
        if 'points' in kwargs:
            colors = ['r', 'g']
            labels = ['Ex7 Keypoints', 'Bootstrapped Keypoints']
            for i in range(len(kwargs['points'])):
                plt.scatter(kwargs['points'][i][0], kwargs['points'][i][1], c=colors[i], s=5, label=labels[i])
            plt.legend()

        plt.draw()
        plt.pause(.1)
