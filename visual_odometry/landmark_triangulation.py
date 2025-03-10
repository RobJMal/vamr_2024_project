import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from cv2.typing import MatLike
from numpy.typing import NDArray
from visual_odometry.common.base_class import BaseClass
from visual_odometry.common.enums.log_level import LogLevel
from visual_odometry.common.params import ParamServer
from visual_odometry.common.utils import Utils
from visual_odometry.common.plot_utils import PlotUtils
from visual_odometry.common.state import Pose, State
from visual_odometry.initialization import Initialization


def checkIter(func):

    def wrapper(self, *args, **kwargs):
        # if self.iter < 2 or "forcePlot" in kwargs:
        return func(self, *args, **kwargs)

    return wrapper


class LandmarkTriangulation(BaseClass):

    def __init__(self, param_server: ParamServer, debug: LogLevel = LogLevel.INFO):
        """
        Given a state with candidate keypoints,
        """
        super().__init__(debug)
        self.iter = 0
        self._init_figures()
        self._info_print("Landmark Triangulation initialized.")

        # Retreive params
        self.params = param_server["landmark_triangulation"]
        self.num_kp: int = self.params['maxNewKeypointsPerIter']
        self.apply_win_thresholding: bool = self.params['applyWindowThresholding']
        self.landmark_angle_threshold = self.params["landmarkAngleThreshold"]
        self.reprojection_error_threshold = self.params["reprojectionErrorThreshold"]
        self.apply_reprojection_error_rejection = self.params["applyReprojectionErrorRejection"]

    @BaseClass.plot_debug
    def _init_figures(self):
        self.vis_figure, self.vis_axs = plt.subplots(2, 2, figsize=(20, 8))
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
        plt.pause(0.5)

    @BaseClass.plot_debug
    def _plot_keypoints_and_landmarks(
        self,
        fig_id: Tuple[int, int],
        ext_pose: Pose,
        P: State.Keypoints,
        X: State.Landmarks,
        candidates=False,
        clear=False,
    ):
        """
        Plots the keypoints and the landmarks.

        P is in the camera frame, X is in the WORLD FRAME
        """
        if clear:
            # Clearing the axes to show changes in landmarks
            self.vis_axs[*fig_id].clear()

        self.vis_axs[*fig_id].set_title("Keypoints and Landmarks")

        # Camera pose and landmarks wrt world frame
        landmarks_wrt_camera = ext_pose @ np.vstack((X, np.ones_like(X[0, :])))
        landmarks_wrt_camera = landmarks_wrt_camera[:3, :]

        # assert np.all(~(landmarks_wrt_camera[2, :] < 0)), "Landmarks for plotting wrt camera frame cannot be behind the camera"

        kp_scaled = PlotUtils._convert_pixels_to_world(P, ext_pose)
        PlotUtils._plot_keypoints_and_landmarks(
            self.vis_axs[*fig_id], ext_pose, kp_scaled, landmarks_wrt_camera, candidates
        )

    def _filter_lost_candidate_keypoints(
        self, curr_image: MatLike, prev_image: MatLike, prev_state: State
    ) -> Tuple[State.Keypoints, State.Keypoints, State.PoseVectors]:
        """
        We have candidate keypoints in the prev_state that correspond to the prev_image.
        Now, find keypoints in the curr_image that correspond to the candidate keypoints.
        The ones that exist will carry over, the ones that were not found will be filtered out.

        """
        C_prev = prev_state.C.T.reshape(-1, 1, 2).astype(np.float32)
        C_new, status, _ = cv2.calcOpticalFlowPyrLK(
            prevImg=prev_image,
            nextImg=curr_image,
            prevPts=C_prev,
            nextPts=None,
            winSize=self.params["winSize"],
            maxLevel=self.params["maxLevel"],
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self.params["criteria_count"],
                self.params["criteria_eps"],
            ),
        )

        # Use RANSAC to estimate essential matrix, E, to filter out outliers
        C_remaining = C_new[status == 1]
        F_remaining = prev_state.F[:, status.ravel() == 1]
        Tau_remaining = prev_state.Tau[:, status.ravel() == 1]

        _, mask = cv2.findEssentialMat(
            C_prev[status == 1], C_remaining, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        C_remaining = C_remaining[mask.ravel() == 1].T
        F_remaining = F_remaining[:, mask.ravel() == 1]
        Tau_remaining = Tau_remaining[:, mask.ravel() == 1]

        self.viz_kp_difference(
            (0, 1),
            C_remaining,
            prev_state.C,
            diff_color="red",
            diff_label="Lost KPs",
            ss_color="purple",
            ss_label="Carried Over KPs",
        )
        self._debug_print(
            f"After filtering, {status[status==1].shape[0]} keypoints from prev frame remain."
        )

        return C_remaining, F_remaining, Tau_remaining

    def _remove_duplicate_candidates_and_add_F_and_Tau(
        self, carried_over_candidates, new_candidates, F, Tau, curr_pose: Pose
    ) -> Tuple[State.Keypoints, State.Keypoints, State.PoseVectors]:
        new_F = np.copy(new_candidates)
        new_Tau = np.tile(curr_pose.reshape((-1, 1)), new_candidates.shape[1])

        all_candidates = np.hstack((carried_over_candidates, new_candidates))
        all_F = np.hstack((F, new_F))
        all_Tau = np.hstack((Tau, new_Tau))
        self._debug_print(
            f"Combining old and new candidates to get: {all_candidates.shape[1]}"
        )

        _, idx = np.unique(all_candidates, return_index=True, axis=1)

        unique_candidates = all_candidates[:, idx]
        unique_F = all_F[:, idx]
        unique_Tau = all_Tau[:, idx]
        self._debug_print(
            f"After deduping, {unique_candidates.shape[1]} candidate keypoints selected to track"
        )
        self.viz_kp_difference(
            (0, 0),
            unique_candidates,
            carried_over_candidates,
            diff_color="green",
            diff_label="New KPs",
            ss_color="purple",
            ss_label="Carried Over KPs",
        )

        return unique_candidates, unique_F, unique_Tau

    def _provide_new_candidate_keypoints(
        self, curr_image: MatLike, prev_image: MatLike, prev_state: State
    ):
        """
        1. Find keypoint correspondances between the old and new images.
        2. Remove the ones that are already there in the filtered candidate keypoints
        """
        kp_curr_raw, _, matches = Initialization.get_keypoints_and_matches(
            curr_image, prev_image
        )

        new_candidates = np.array(
            [kp_curr_raw[m.queryIdx].pt for m in matches], dtype=np.float32
        ).T

        # Remove candidate keypoints that are sufficiently close to existing keypoints in the state
        if prev_state.P is not None:
            dist = cdist(prev_state.P.T, new_candidates.T, metric="cityblock")
            min_dist = np.min(dist, axis=0)
            new_candidates = new_candidates[:, min_dist >
                                            self.params["equalKeypointThreshold"]]

        self._debug_print(
            f"Found: {new_candidates.shape[1]} new candidate keypoints")

        if self.apply_win_thresholding:
            return self._select_n_new_keypoints(curr_image.shape, new_candidates)

        return new_candidates

    def _select_n_new_keypoints(self, img_size: Tuple[int, int], candidates: State.Keypoints):
        if candidates.shape[1] <= self.num_kp:
            return candidates

        self._debug_print(f"Eligible new keypoints: {candidates.shape}")

        num_wins = int(np.sqrt(self.num_kp))
        x_bins = np.linspace(0, img_size[0], num_wins+1)
        y_bins = np.linspace(0, img_size[1], num_wins+1)

        # Associate each keypoint with a window ID
        # windows = np.zeros((1, candidates.shape[1]))

        # naively:
        x_id = np.digitize(candidates, x_bins) - 1
        y_id = np.digitize(candidates, y_bins) - 1

        _, idx = np.unique(np.vstack((x_id, y_id)), axis=1, return_index=True)

        candidates = candidates[:, idx]
        self._debug_print(
            f"After windowing, new keypoints: {candidates.shape}")
        return candidates

    @staticmethod
    def _inv_homo_transform(T):
        R = T[:3, :3]
        t = T[:3, 3][:, None]

        return np.block([[R.T, -R.T @ t], [0, 0, 0, 1]])

    @staticmethod
    def _get_extrinsic_from_pose(T: NDArray):
        R = T[:3, :3]
        t = T[:3, 3][:, None]
        return R, t

    def _get_reprojection_error_mask(self, C, proj_mat, X) -> NDArray:
        """
        Return a boolean mask of selected landmarks to reject
        """
        if not self.apply_reprojection_error_rejection:
            return np.ones(C.shape[1], dtype=np.bool)
        reprojected = proj_mat @ np.vstack((X, np.ones(X.shape[1])))
        reprojected = reprojected[:2, :] / reprojected[2, :]
        error = np.linalg.norm(reprojected - C, axis=0)

        mask = error < self.reprojection_error_threshold
        self._debug_print(
            f"Reprojection Error Mask Rejects: {np.sum(~mask)}/{mask.shape[0]} keypoints")

        self._plot_reproj_error_summary((1, 0), error, mask)
        return mask

    def _get_squared_sum_reprojection_residual(self, x0, proj_mat_f, proj_mat_c, F, C):
        X = x0.reshape((3, -1))

        reproj_f = proj_mat_f @ np.vstack((X, np.ones(X.shape[1])))
        reproj_f = reproj_f[:2, :] / reproj_f[2, :]
        res_f = np.linalg.norm(reproj_f - F, axis=0) ** 2

        reproj_c = proj_mat_c @ np.vstack((X, np.ones(X.shape[1])))
        reproj_c = reproj_c[:2, :] / reproj_c[2, :]
        res_c = np.linalg.norm(reproj_c - C, axis=0) ** 2

        res_z = np.zeros_like(res_c)  # Dummy

        return res_c + res_f  # np.hstack((res_f, res_c, res_z))

    def _refine_triangulation_with_lm(self, proj_mat_f, proj_mat_c, F, C, X_init):
        """
        Improve standard SVD based triangulation with LM optimization based on the reprojection error.

        This d
        """
        # First remove points that have a high reprojection error to begin with
        x0 = X_init.reshape(-1)

        # reproj_error = np.linalg.norm(self._get_squared_sum_reprojection_residual(x0, proj_mat_f, proj_mat_c, F, C).reshape((3, -1)), axis=0)
        reproj_residual = self._get_squared_sum_reprojection_residual(
            x0, proj_mat_f, proj_mat_c, F, C)
        filter_mask = reproj_residual < self.reprojection_error_threshold
        self._debug_print(
            f"Removed outlier landmarks with high reprojection error before minimization. Using {np.sum(filter_mask)}/{X_init.shape[1]}")

        x0 = X_init[:, filter_mask].reshape(-1)
        F = F[:, filter_mask]
        C = C[:, filter_mask]

        reproj_error = np.linalg.norm(self._get_squared_sum_reprojection_residual(
            x0, proj_mat_f, proj_mat_c, F, C), ord=1)
        self._debug_print(
            f"Reprojection Error before minimization {reproj_error}")
        out_x = least_squares(self._get_squared_sum_reprojection_residual, x0, args=(
            proj_mat_f, proj_mat_c, F, C), method='trf')
        reproj_error = np.linalg.norm(self._get_squared_sum_reprojection_residual(
            out_x.x, proj_mat_f, proj_mat_c, F, C), ord=1)
        self._debug_print(
            f"Reprojection Error after minimization {reproj_error}")
        X_refined = out_x.x.reshape((3, -1))

        X_init[:, ~filter_mask] = 0
        X_init[:, filter_mask] = X_refined

        return X_refined, filter_mask

    def _get_landmarks_for_keypoints(
        self,
        K: NDArray,
        updated_state: State,
        C: State.Keypoints,
        F: State.Keypoints,
        Tau: State.PoseVectors,
        curr_pose: Pose,
    ) -> Tuple[
        State.Keypoints,
        State.Landmarks,
        State.Keypoints,
        State.Keypoints,
        State.PoseVectors,
    ]:
        """
        For the candidate keypoints, find out the landmarks that meet the threshold for getting added to the main queue,

        @return: P_new, X_new, C_remaining, F_remaining, Tau_remaining
        """
        curr_R_ext, curr_t_ext = self._get_extrinsic_from_pose(curr_pose)
        proj_mat_curr: NDArray = K @ np.block([curr_R_ext, curr_t_ext])
        points_world = np.zeros((3, C.shape[1]), dtype=np.float32)
        angles = np.zeros(C.shape[1], dtype=np.float32)
        angles[:] = np.nan

        for cnt, (cs, fs, taus) in enumerate(zip(C.T, F.T, Tau.T)):
            group_idx = np.array([cnt])
            taus = taus[:, None]
            fs = fs[:, None]
            cs = cs[:, None]

            tau: NDArray = taus.reshape((4, 4))
            proj_mat_f: NDArray = K @ tau[:3, :]

            pXs_C_wrt_w_4D: NDArray = cv2.triangulatePoints(
                proj_mat_f, proj_mat_curr, fs, cs
            )
            pXs_C_wrt_w = pXs_C_wrt_w_4D[:3, :] / pXs_C_wrt_w_4D[3, :]
            # These are multiple landmarks triangulated from the same pose
            pXs_C_wrt_f = tau @ np.vstack((pXs_C_wrt_w,
                                          np.ones(pXs_C_wrt_w.shape[1])))
            pXs_C_wrt_c = curr_pose @ np.vstack(
                (pXs_C_wrt_w, np.ones(pXs_C_wrt_w.shape[1])))

            rear_triangulated_pts = (pXs_C_wrt_f[2] < 0) | (pXs_C_wrt_c[2] < 0)
            pXs_C_wrt_w = pXs_C_wrt_w[:, ~rear_triangulated_pts]
            group_idx = group_idx[~rear_triangulated_pts]

            if pXs_C_wrt_w.size == 0:
                continue

            # pXs_C_wrt_w, accepted_pts = self._refine_triangulation_with_lm(
            #     proj_mat_f, proj_mat_curr, fs[:, ~rear_triangulated_pts], cs[:, ~rear_triangulated_pts], pXs_C_wrt_w)
            # if accepted_pts.size == 0:
            #     continue
            # group_idx = group_idx[accepted_pts]

            pXs_C_wrt_f = tau @ np.vstack((pXs_C_wrt_w,
                                          np.ones(pXs_C_wrt_w.shape[1])))
            pXs_C_wrt_c = curr_pose @ np.vstack(
                (pXs_C_wrt_w, np.ones(pXs_C_wrt_w.shape[1])))
            dot_products = np.sum(np.multiply(
                pXs_C_wrt_f, pXs_C_wrt_c), axis=0)
            alpha = np.arccos(
                dot_products
                / (np.linalg.norm(pXs_C_wrt_c, axis=0) * np.linalg.norm(pXs_C_wrt_f, axis=0))
            )
            assert np.all(np.isnan(angles[group_idx])
                          ), "Overwriting Initialized Angles"
            angles[group_idx] = alpha
            points_world[:, group_idx] = pXs_C_wrt_w

        X_curr_camera = Utils.homogenous_mat_mult(curr_pose, updated_state.X)
        points_camera = Utils.homogenous_mat_mult(curr_pose, points_world)
        outlier_rejection_mask = self._get_outlier_rejection_mask(
            X_curr_camera, points_camera, self.params["outlierRejectionZThreshold"])
        points_world[:, outlier_rejection_mask] = 0
        angles[outlier_rejection_mask] = 0

        self._debug_print(
            f"After removing outliers in the X and Z axis, we have {np.sum(outlier_rejection_mask)}/{points_world.shape[1]} landmarks available")

        self._plot_keypoints_and_landmarks(
            (1, 1), curr_pose, C, points_world, candidates=True, clear=False
        )

        angles[np.isnan(angles)] = 0.0
        self._info_print(
            f"Angles: max: {np.max(angles)}, min: {np.min(angles)}")
        eligible_keypoint_mask = angles > self.landmark_angle_threshold

        reprojection_error_mask = self._get_reprojection_error_mask(
            C, proj_mat_curr, points_world)

        selection_mask = eligible_keypoint_mask & reprojection_error_mask

        P_new = C[:, selection_mask]
        X_new = points_world[:, selection_mask]

        C_remain = C[:, ~selection_mask]
        F_remain = F[:, ~selection_mask]
        Tau_remain = Tau[:, ~selection_mask]

        # rank the remaining candidates based on dist from existing and angle
        angle_rank = np.argsort(angles[selection_mask])
        distance_rank = np.argsort(
            np.mean(cdist(updated_state.P.T, P_new.T, metric="cityblock")))

        num_kp_to_add = max(
            0, self.params["maxNumLandmarks"] - updated_state.P.shape[1])

        num_kp_to_add = np.clip(num_kp_to_add, None,
                                self.params["limitNewLandmarks"])

        rank = np.argsort(
            angle_rank + distance_rank)[:num_kp_to_add]

        P_new = P_new[:, rank]
        X_new = X_new[:, rank]

        self._info_print(
            f"Found {np.sum(eligible_keypoint_mask)} new landmarks to add to the queue. {np.sum(~eligible_keypoint_mask)} candidates remain."
        )

        return P_new, X_new, C_remain, F_remain, Tau_remain

    @staticmethod
    def _get_outlier_rejection_mask(X_curr, X_new, threshold):
        """
        From a set of landmarks, X_new;
        Remove outliers based on a stddev threshold from statistics of the current landmarks X_curr
        """
        mean = np.mean(X_curr, axis=1)[:, None]
        stddev = np.std(X_curr, axis=1)[:, None]

        zscore_all_dim = np.abs((X_new - mean) / stddev)
        zscore_avg = np.average(zscore_all_dim, axis=0)

        reject_mask = zscore_avg > threshold
        return reject_mask

    @BaseClass.plot_debug
    def _plot_reproj_error_summary(self, fig_id: Tuple[int, int], error, mask):
        x_ax = np.arange(error.shape[0])
        self.vis_axs[*fig_id].scatter(x_ax[mask], error[mask],
                                      color="Green", label=f"Accepted: {np.sum(mask)}", alpha=0.1)
        self.vis_axs[*fig_id].scatter(x_ax[~mask], error[~mask],
                                      color="Red", label=f"Rejected: {np.sum(~mask)}", alpha=0.1)
        self.vis_axs[*fig_id].set_title("Reprojection Error")

    def perform_triangulation(
        self,
        K: NDArray,
        curr_image: MatLike,
        prev_image: MatLike,
        updated_state: State,
        prev_state: State,
        curr_pose: Pose,
    ):
        if prev_state.C.size == 0:
            # Empty the first time
            new_candidates = self._provide_new_candidate_keypoints(
                curr_image, prev_image, prev_state)
            C_new, F_new, Tau_new = self._remove_duplicate_candidates_and_add_F_and_Tau(
                np.empty((2, 0)), new_candidates, np.empty(
                    (2, 0)), np.empty((16, 0)), curr_pose
            )

            return np.empty((2, 0)), np.empty((3, 0)), C_new, F_new, Tau_new

        # Filter Lost Candidates First
        self._debug_print(
            f"Prev state number of candidates: {prev_state.C.shape[1]}")
        C_filtered, F_filtered, Tau_filtered = self._filter_lost_candidate_keypoints(
            curr_image, prev_image, prev_state
        )

        # Triangulate the points from the current candidates and evaluate the ones to Remove
        P_new, X_new, C_remain, F_remain, Tau_remain = (
            self._get_landmarks_for_keypoints(
                K, updated_state, C_filtered, F_filtered, Tau_filtered, curr_pose
            )
        )

        # Get new candidates from the current and previous frame
        self.viz_image(
            (0, 0),
            curr_image,
            cmap="gray",
            label="Current Image",
            title="Evaluated Candidates",
        )
        new_candidates = self._provide_new_candidate_keypoints(
            curr_image, prev_image, prev_state)
        C_new, F_new, Tau_new = self._remove_duplicate_candidates_and_add_F_and_Tau(
            C_remain, new_candidates, F_remain, Tau_remain, curr_pose
        )

        return P_new, X_new, C_new, F_new, Tau_new

    def __call__(
        self,
        K: NDArray,
        curr_image: MatLike,
        prev_image: MatLike,
        updated_state: State,
        prev_state: State,
        curr_pose: Pose,
    ) -> State:
        """
        There are three steps here:
            1. Find which candidate keypoints are still there in the new image.
            1. Find keypoints from the image and add it to candidate keypoints, their initial pose, and the pose we found of the camera at the time for storage.
            2. Track them for some length. The keypoints that have been tracked for that much time or more have to be popped for analysis I guess...?
            3.
        """

        self._clear_figures()
        # self.viz_curr_and_prev_img(curr_image, prev_image)
        self.K = K

        self._plot_keypoints_and_landmarks(
            (1, 1),
            curr_pose,
            updated_state.P,
            updated_state.X,
            candidates=False,
            clear=True,
        )

        P_new, X_new, C_new, F_new, Tau_new = self.perform_triangulation(
            K, curr_image, prev_image, updated_state, prev_state, curr_pose
        )

        updated_state.P = np.hstack((updated_state.P, P_new))
        updated_state.X = np.hstack((updated_state.X, X_new))
        updated_state.C = C_new
        updated_state.F = F_new
        updated_state.Tau = Tau_new

        self._refresh_figures()

        self.iter += 1

        return updated_state

    @ checkIter
    def get_axs(self, *args, **kwargs):
        if "forcePlot" in kwargs:
            del kwargs["forcePlot"]
        return super().get_axs(*args, **kwargs)

    @ BaseClass.plot_debug
    @ checkIter
    def viz_curr_and_prev_img(self, curr_image, prev_image) -> None:
        self.vis_axs[0, 0].imshow(prev_image)
        self.vis_axs[0, 0].set_title("Previous Image")
        self.vis_axs[0, 1].imshow(curr_image)
        self.vis_axs[0, 1].set_title("Current Image")

    @ BaseClass.plot_debug
    @ checkIter
    def viz_image(
        self,
        fig_idx: Tuple[int, int],
        image: MatLike,
        cmap: str,
        label: str,
        title=None,
        **kwargs,
    ):
        self.vis_axs[*fig_idx].imshow(image, cmap=cmap, label=label)
        if title:
            self.vis_axs[*fig_idx].set_title(title)

    @ BaseClass.plot_debug
    @ checkIter
    def viz_keypoints(
        self, fig_idx: Tuple[int, int], kps: NDArray, color: str, label: str, **kwargs
    ):
        if "forcePlot" in kwargs:
            del kwargs["forcePlot"]
        self.vis_axs[*fig_idx].scatter(
            kps[0, :],
            kps[1, :],
            color=color,
            label=f"{label}: {kps.shape[1]}",
            s=1.0,
            **kwargs,
        )

    @ BaseClass.plot_debug
    @ checkIter
    def viz_kp_difference(
        self,
        fig_idx: Tuple[int, int],
        superset: State.Keypoints,
        subset: State.Keypoints,
        diff_color: str,
        diff_label: str,
        ss_color: str,
        ss_label: str,
    ):
        """
        Visualize the set difference between superset and subset, and also the subset (in separate colors)
        """
        # The masking logic is wrapped here to reduce this computation if plotting is disabled
        diff_mask = ~np.any(
            np.all(superset[:, :, None] == subset[:, None, :], axis=0), axis=1
        )
        diff_kps = superset[:, diff_mask]
        self.viz_keypoints(fig_idx, diff_kps, diff_color, diff_label)
        self.viz_keypoints(fig_idx, subset, ss_color, ss_label)
