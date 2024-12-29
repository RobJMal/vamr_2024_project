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


class LandmarkTriangulation(BaseClass):

    def __init__(self, param_server: ParamServer, debug: LogLevel = LogLevel.INFO):
        """
        Given a state with candidate keypoints, 
        """
        super().__init__(debug)
        self._init_figures()
        self._info_print("Landmark Triangulation initialized.")

        # Retreive params
        self.params = param_server["landmark_triangulation"]

    @BaseClass.plot_debug
    def _init_figures(self):
        self.vis_figure, self.vis_axs = plt.subplots(3, 2, figsize=(14, 14))
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

    def __call__(self, curr_image: MatLike, prev_image: MatLike, prev_state: State, curr_pose: Pose):
        """
        There are three steps here:
            1. Find which candidate keypoints are still there in the new image.
            1. Find keypoints from the image and add it to candidate keypoints, their initial pose, and the pose we found of the camera at the time for storage.
            2. Track them for some length. The keypoints that have been tracked for that much time or more have to be popped for analysis I guess...?
            3. 
        """
        self._clear_figures()
        state = deepcopy(prev_state)
        self.viz_curr_and_prev_img(curr_image, prev_image)
        self.viz_image((1, 0), curr_image, cmap="gray", label="Current Image", title="Lost KPs")
        self._debug_print(f"Prev state number of candidates: {prev_state.C.shape[1]}")
        C_remaining, F_remaining, Tau_remaining = self._filter_lost_candidate_keypoints(curr_image, prev_image, prev_state)

        # Now I need to add more keypoints based on the prev image and current image. I guess I can find keypoints once again, filter out the ones that are in the candidates now
        self.viz_image((1, 1), curr_image, cmap="gray", label="Current Image", title="Evaluated Candidates")
        new_candidates = self._provide_new_candidate_keypoints(curr_image, prev_image)
        C_new, F_new, Tau_new = self._remove_duplicate_candidates_and_add_F_and_Tau(C_remaining, new_candidates, F_remaining, Tau_remaining, curr_pose)
        
        self._refresh_figures()

        state.C = C_new
        state.F = F_new
        state.Tau = Tau_new

        return state


    @BaseClass.plot_debug
    def viz_curr_and_prev_img(self, curr_image, prev_image) -> None:
        self.vis_axs[0, 0].imshow(prev_image)
        self.vis_axs[0, 0].set_title("Previous Image")
        self.vis_axs[0, 1].imshow(curr_image)
        self.vis_axs[0, 1].set_title("Current Image")

    @BaseClass.plot_debug
    def viz_image(self, fig_idx: Tuple[int, int], image: MatLike, cmap: str, label: str, title=None):
        self.vis_axs[*fig_idx].imshow(image, cmap=cmap, label=label)
        if title:
            self.vis_axs[*fig_idx].set_title(title)

    @BaseClass.plot_debug
    def viz_keypoints(self, fig_idx: Tuple[int, int], kps: NDArray, color: str, label: str):
        self.vis_axs[*fig_idx].scatter(kps[0, :], kps[1, :], color=color, label=f"{label}: {kps.shape[1]}", s=0.5)

    @BaseClass.plot_debug
    def viz_kp_difference(self, fig_idx: Tuple[int, int], superset: State.Keypoints, subset: State.Keypoints, diff_color: str, diff_label: str, ss_color: str, ss_label: str):
        """
        Visualize the set difference between superset and subset, and also the subset (in separate colors)
        """
        # The masking logic is wrapped here to reduce this computation if plotting is disabled
        diff_mask = ~np.any(np.all(superset[:, :, None] == subset[:, None, :], axis=0), axis=1)
        diff_kps = superset[:, diff_mask]
        self.viz_keypoints(fig_idx, diff_kps, diff_color, diff_label)
        self.viz_keypoints(fig_idx, subset, ss_color, ss_label)
