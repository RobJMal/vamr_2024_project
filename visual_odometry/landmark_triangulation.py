import cv2
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict, Tuple
from cv2.typing import MatLike
from numpy.typing import NDArray
from visual_odometry.common.base_class import BaseClass
from visual_odometry.common.enums.log_level import LogLevel
from visual_odometry.common.params import ParamServer
from visual_odometry.common.state import Pose, State
from visual_odometry.initialization import Initialization

def checkIter(func):

    def wrapper(self, *args, **kwargs):
        if self.iter < 2 or "forcePlot" in kwargs:
            return func(self, *args, **kwargs)

    return wrapper


class LandmarkTriangulation(BaseClass):

    def __init__(self, param_server: ParamServer, debug: LogLevel = LogLevel.INFO):
        """
        Given a state with candidate keypoints,
        """
        super().__init__(debug)
        self.iter = 0
        self.THRESH = {5, 10, 15}
        self._init_figures()
        self._info_print("Landmark Triangulation initialized.")

        # Retreive params
        self.params = param_server["landmark_triangulation"]

    @BaseClass.plot_debug
    def _init_figures(self):
        self.vis_figure, self.vis_axs = plt.subplots(3, 2, figsize=(20, 10))
        self.vis_figure.suptitle("Landmark Triangulation Visualization")

    def _get_axs_impl(self, fig_idx):
        return self.vis_axs[fig_idx]

    @BaseClass.plot_debug
    def _clear_figures(self):
        for ax in self.vis_axs.flat:
            ax.clear()

    @BaseClass.plot_debug
    def _refresh_figures(self):
        self.vis_figure.canvas.draw_idle()
        for ax in self.vis_axs.flat:
            ax.legend()
        plt.pause(.1)

    def _filter_lost_candidate_keypoints(self, curr_image: MatLike, prev_image: MatLike, prev_state: State) -> Tuple[State.Keypoints, State.Keypoints, State.PoseVectors]:
        """
        We have candidate keypoints in the prev_state that correspond to the prev_image.
        Now, find keypoints in the curr_image that correspond to the candidate keypoints.
        The ones that exist will carry over, the ones that were not found will be filtered out.

        """
        C_prev = prev_state.C.T.reshape(-1, 1, 2).astype(np.float32)
        C_new, status, _ = cv2.calcOpticalFlowPyrLK(prevImg = prev_image,
                                                    nextImg = curr_image,
                                                    prevPts = C_prev,
                                                    nextPts = None,
                                                    winSize = self.params["winSize"],
                                                    maxLevel = self.params["maxLevel"],
                                                    criteria = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.params["criteria_count"], self.params["criteria_eps"]))


        # get the matching points
        C_remaining = C_new[status == 1].T
        F_remaining = prev_state.F[:, status.ravel()==1]
        Tau_remaining = prev_state.Tau[:, status.ravel()==1]

        self.viz_kp_difference((1, 0), C_remaining, prev_state.C, diff_color="red", diff_label="Lost KPs", ss_color="purple", ss_label="Carried Over KPs")
        self._debug_print(f"After filtering, {status[status==1].shape[0]} keypoints from prev frame remain.")

        return C_remaining, F_remaining, Tau_remaining

    def _remove_duplicate_candidates_and_add_F_and_Tau(self, carried_over_candidates, new_candidates, F, Tau, curr_pose: Pose) -> Tuple[State.Keypoints, State.Keypoints, State.PoseVectors]:
        new_F = np.copy(new_candidates)
        new_Tau = np.tile(curr_pose.reshape((-1, 1)), new_candidates.shape[1])

        all_candidates = np.hstack((carried_over_candidates, new_candidates))
        all_F = np.hstack((F, new_F))
        all_Tau = np.hstack((Tau, new_Tau))
        self._info_print(f"Combining old and new candidates to get: {all_candidates.shape[1]}")

        _, idx = np.unique(all_candidates, return_index=True, axis=1)

        unique_candidates = all_candidates[:, idx]
        unique_F = all_F[:, idx]
        unique_Tau = all_Tau[:, idx]
        self._info_print(f"After deduping, {unique_candidates.shape[1]} candidate keypoints selected to track")
        self.viz_kp_difference((1, 1), unique_candidates, carried_over_candidates, diff_color="green", diff_label="New KPs", ss_color="purple", ss_label="Carried Over KPs")

        return unique_candidates, unique_F, unique_Tau

    def _provide_new_candidate_keypoints(self, curr_image: MatLike, prev_image: MatLike):
        """
        1. Find keypoint correspondances between the old and new images.
        2. Remove the ones that are already there in the filtered candidate keypoints
        """
        kp_curr_raw, _, matches = Initialization.get_keypoints_and_matches(curr_image, prev_image)

        new_candidates = np.array([kp_curr_raw[m.queryIdx].pt for m in matches], dtype=np.float32).T
        self._debug_print(f"Found: {new_candidates.shape[1]} new candidate keypoints")
        return new_candidates

    # def _visualization_stuff_ignore(self, curr_image: MatLike, prev_image: MatLike, prev_state: State, curr_pose: Pose) -> Tuple[State.Keypoints, State.Keypoints, State.PoseVectors]:

    #     # # Visualize C_new and F_new, gradually we would like to see a drift here.
    #     # self.viz_image((2, 1), curr_image, cmap="gray", label="Current Image", title="Final Candidates", forcePlot=True)
    #     # # idx = np.random.randint(0, C_new.shape[1], size=np.clip(10, None, C_new.shape[1]))
    #     # idx = np.arange(10)
    #     # f_pts = F_new[:, idx]
    #     # c_pts = C_new[:, idx]
    #     # self.viz_keypoints((2, 1), f_pts, color="orange", label="F_new", alpha=0.5, forcePlot=True)
    #     # self.viz_keypoints((2, 1), c_pts, color="red", label="C_new", alpha=0.5, forcePlot=True)

    #     # for pt1, pt2 in zip(f_pts.T, c_pts.T):
    #     #     X = [pt1[0], pt2[0]]
    #     #     Y = [pt1[1], pt2[1]]
    #     #     self.get_axs((2, 1), forcePlot=True).plot(X, Y)  # Plot all lines at once

    @staticmethod
    def _inv_homo_transform(T):
        R = T[:3, :3]
        t = T[:3, 3][:, None]

        return np.block([
            [R.T, -R.T @ t],
            [0, 0, 0, 1]
            ])

    @staticmethod
    def _get_extrinsic_from_pose(T: NDArray):
        R = T[:3, :3]
        t = T[:3, 3][:, None]

        return R, t

    def _triangulate_points(self, K, first_kp: State.Keypoints, first_pose: Pose, curr_kp: State.Keypoints, curr_R_ext: NDArray, curr_t_ext: NDArray, proj_mat_curr: NDArray) -> Tuple[bool, NDArray, float]:
        # Triangulate points
        first_R_ext, first_t_ext = self._get_extrinsic_from_pose(first_pose)
        proj_mat_first = K @ np.block([first_R_ext, first_t_ext])

        pX_C_world_4D = cv2.triangulatePoints(proj_mat_first, proj_mat_curr, first_kp.T, curr_kp.T)
        pX_C_world_3D = pX_C_world_4D[:3, :] / pX_C_world_4D[3, :] # This is in the world frame.

        # Convert point to camera frame (vector from center of the camera to the point).
        # If the z axis is negative, the point is behind the camera and thus, has to be rejected
        pX_C_camCurr_3D: NDArray = curr_R_ext.T @ pX_C_world_3D - curr_R_ext.T @ curr_t_ext
        pX_C_camFirst_3D: NDArray = first_R_ext.T @ pX_C_world_3D - first_R_ext.T @ first_t_ext

        if pX_C_camCurr_3D[2] < 0 or pX_C_camFirst_3D[2] < 0:
            self._debug_print(f"Projection behind the camera - z1: {pX_C_camCurr_3D[2] < 0} z2: {pX_C_camFirst_3D[2] < 0}")
            return False, np.zeros(3), 0.0

        alpha = np.acos(pX_C_camFirst_3D.T.dot(-pX_C_camCurr_3D) / (np.linalg.norm(pX_C_camCurr_3D) * np.linalg.norm(pX_C_camCurr_3D)))

        return True, pX_C_world_3D, alpha

    def _get_new_landmarks(self, K: NDArray, C_new: State.Keypoints, F_new: State.Keypoints, Tau_new: State.PoseVectors, curr_pose: Pose):
        curr_R_ext, curr_t_ext = self._get_extrinsic_from_pose(curr_pose)
        proj_mat_curr = K @ np.block([curr_R_ext, curr_t_ext])
        points = []
        angles = []
        for c, f, tau in zip(C_new.T, F_new.T, Tau_new.T):
            tau = tau.reshape((4, 4))
            success, point, alpha = self._triangulate_points(K, f, tau, c, curr_R_ext, curr_t_ext, proj_mat_curr)
            if not success:
                continue
            points.append(point)
            angles.append(alpha)

        angles = np.array(angles).ravel()
        angles = angles[~np.isnan(angles)]

        print(f"Max Angle: {np.max(angles)} Min Angle: {np.min(angles)}")

    def perform_triangulation(self, K: NDArray, curr_image: MatLike, prev_image: MatLike, prev_state: State, curr_pose: Pose):

        # Filter Lost Candidates First
        self.viz_image((1, 0), curr_image, cmap="gray", label="Current Image", title="Lost KPs")
        self._debug_print(f"Prev state number of candidates: {prev_state.C.shape[1]}")
        C_remaining, F_remaining, Tau_remaining = self._filter_lost_candidate_keypoints(curr_image, prev_image, prev_state)

        # Triangulate the points from the current candidates and evaluate the ones to Remove
        self._get_new_landmarks(K, C_remaining, F_remaining, Tau_remaining, curr_pose)

        # Get new candidates from the current and previous frame
        self.viz_image((1, 1), curr_image, cmap="gray", label="Current Image", title="Evaluated Candidates")
        new_candidates = self._provide_new_candidate_keypoints(curr_image, prev_image)
        C_new, F_new, Tau_new = self._remove_duplicate_candidates_and_add_F_and_Tau(C_remaining, new_candidates, F_remaining, Tau_remaining, curr_pose)

        return C_new, F_new, Tau_new

    def __call__(self, K: NDArray, curr_image: MatLike, prev_image: MatLike, prev_state: State, curr_pose: Pose):
        """
        There are three steps here:
            1. Find which candidate keypoints are still there in the new image.
            1. Find keypoints from the image and add it to candidate keypoints, their initial pose, and the pose we found of the camera at the time for storage.
            2. Track them for some length. The keypoints that have been tracked for that much time or more have to be popped for analysis I guess...?
            3.
        """
        self._clear_figures()
        self.viz_curr_and_prev_img(curr_image, prev_image)

        state = deepcopy(prev_state)


        # C_new, F_new, Tau_new = self._track_candidates(curr_image, prev_image, prev_state, curr_pose)
        C_new, F_new, Tau_new = self.perform_triangulation(K, curr_image, prev_image, prev_state, curr_pose)

        self._refresh_figures()

        # Figure out what to return from here...
        state.C = C_new
        state.F = F_new
        state.Tau = Tau_new
        self.iter += 1

        return state

    @checkIter
    def get_axs(self, *args, **kwargs):
        if "forcePlot" in kwargs:
            del kwargs["forcePlot"]
        return super().get_axs(*args, **kwargs)

    @BaseClass.plot_debug
    @checkIter
    def viz_curr_and_prev_img(self, curr_image, prev_image) -> None:
        self.vis_axs[0, 0].imshow(prev_image)
        self.vis_axs[0, 0].set_title("Previous Image")
        self.vis_axs[0, 1].imshow(curr_image)
        self.vis_axs[0, 1].set_title("Current Image")

    @BaseClass.plot_debug
    @checkIter
    def viz_image(self, fig_idx: Tuple[int, int], image: MatLike, cmap: str, label: str, title=None, **kwargs):
        self.vis_axs[*fig_idx].imshow(image, cmap=cmap, label=label)
        if title:
            self.vis_axs[*fig_idx].set_title(title)

    @BaseClass.plot_debug
    @checkIter
    def viz_keypoints(self, fig_idx: Tuple[int, int], kps: NDArray, color: str, label: str, **kwargs):
        if "forcePlot" in kwargs:
            del kwargs["forcePlot"]
        self.vis_axs[*fig_idx].scatter(kps[0, :], kps[1, :], color=color, label=f"{label}: {kps.shape[1]}", s=1.0, **kwargs)

    @BaseClass.plot_debug
    @checkIter
    def viz_kp_difference(self, fig_idx: Tuple[int, int], superset: State.Keypoints, subset: State.Keypoints, diff_color: str, diff_label: str, ss_color: str, ss_label: str):
        """
        Visualize the set difference between superset and subset, and also the subset (in separate colors)
        """
        # The masking logic is wrapped here to reduce this computation if plotting is disabled
        diff_mask = ~np.any(np.all(superset[:, :, None] == subset[:, None, :], axis=0), axis=1)
        diff_kps = superset[:, diff_mask]
        self.viz_keypoints(fig_idx, diff_kps, diff_color, diff_label)
        self.viz_keypoints(fig_idx, subset, ss_color, ss_label)
