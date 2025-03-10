from typing import Optional, Sequence, Tuple
from cv2.typing import MatLike
import numpy as np
import os
import glob
import cv2
import time
import matplotlib.pyplot as plt
import argparse

from numpy.typing import NDArray

from visual_odometry.common import ParamServer
from visual_odometry.common import BaseClass
from visual_odometry.common.enums import LogLevel
from visual_odometry.common.enums import DataSet
from visual_odometry.common import State
from visual_odometry.common.plot_utils import PlotUtils
from visual_odometry.common.state import Pose
from visual_odometry.initialization import Initialization
from visual_odometry.keypoint_tracking import KeypointTracker
from visual_odometry.landmark_triangulation import LandmarkTriangulation
from visual_odometry.pose_estimating import PoseEstimator


def parse_args():
    """Argument parses for command line arguments."""

    parser = argparse.ArgumentParser(description='Visual Odometry pipeline')
    parser.add_argument('--dataset', type=str, default='KITTI',
                        help='Dataset to run the pipeline on')

    parser.add_argument('--debug', type=str, default='INFO',
                        help='Debug level: NONE INFO, DEBUG, VISUALIZATION')

    parser.add_argument('--params', type=str, default='params/pipeline_params.yaml',
                        help='Path to the parameters file')

    parser.add_argument('--no-bootstrap', action='store_false',
                        help='Do not use bootstrap, initialize based on given keypoints')

    return parser.parse_args()


class VisualOdometryPipeline(BaseClass):
    def __init__(self, param_server: ParamServer, debug: LogLevel = LogLevel.INFO):
        super().__init__(debug)
        self._param_server = param_server

        self._dataset_paths = {"KITTI": os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets/kitti"),
                               "MALAGA": os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets/malaga-urban-dataset-extract-07"),
                               "PARKING": os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets/parking")}

        self._init_figures()
        self.params = self._param_server["main"]
        self.left_images: Optional[list[str]] = None # Will be None if dataset is not Malaga

        self.initialization = Initialization(param_server=self._param_server, debug=self.debug)
        self.keypoint_tracker = KeypointTracker(param_server=self._param_server, debug=self.debug)
        self.pose_estimator = PoseEstimator(param_server=self._param_server, debug=self.debug)
        self.landmark_triangulation = LandmarkTriangulation(param_server=self._param_server, debug=self.debug)
        self.keyframe_frequency = self.params["keyframeFreq"]
        self.poses = []
        self.states = []

        self._info_print(
            f"VO monocular pipeline initialized\n - Log level: {debug.name}\n - ParamServer: {self._param_server}")

    def _get_dataset_metadata(self, dataset: DataSet) -> None:
        # Setup
        self.ground_truth = None
        match dataset:
            case DataSet.KITTI:
                ground_truth = np.loadtxt(
                    self._dataset_paths["KITTI"] + '/poses/05.txt')
                self.ground_truth = ground_truth[:, [-9, -1]]
                self.last_frame: int = 2760
                self.K: NDArray = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                             [0, 7.188560000000e+02, 1.852157000000e+02],
                             [0, 0, 1]])
            case DataSet.MALAGA:
                images_path = os.path.join(
                    self._dataset_paths["MALAGA"], 'malaga-urban-dataset-extract-07_rectified_800x600_Images')
                images = sorted(glob.glob(os.path.join(images_path, '*')))

                self.left_images = images[2::2]
                self.last_frame: int = len(self.left_images) - 1

                self.K: NDArray = np.array([[621.18428, 0, 404.0076],
                              [0, 621.18428, 309.05989],
                              [0, 0, 1]])
                
                self.ground_truth = None
            case DataSet.PARKING:
                self.last_frame: int = 598
                self.K: NDArray = np.loadtxt(
                    self._dataset_paths["PARKING"] + '/K.txt', delimiter=',')
                ground_truth = np.loadtxt(
                    self._dataset_paths["PARKING"] + '/poses.txt')
                self.ground_truth = ground_truth[:, [-9, -1]]

    def _get_kitti_debug_points(self) -> Tuple[State, Sequence[int], NDArray]:
        state = State()
        state.P = np.loadtxt(os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets/kitti/kp_for_debug.txt")).T
        state.P[[0,1]] = state.P[[1,0]] # swap x and y
        from_index = 1
        to_index = self.last_frame + 1
        img0_path = os.path.join(
                self._dataset_paths["KITTI"], '05/image_0/000000.png')
        img0 = np.array(cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE))
        prev_img = img0

        # TODO: this should be removed in the final version
        plt.figure()
        plt.imshow(img0, cmap='gray')
        plt.scatter(state.P[0], state.P[1], c='r', s=5)
        plt.title("Initial keypoints")
        plt.show()

        return state, range(from_index, to_index), prev_img

    def _init_dataset(self, dataset: DataSet, use_bootstrap: bool) -> Tuple[State, Sequence[int], NDArray]:
        self._get_dataset_metadata(dataset)
        self.dataset = dataset

        #  Bootstrap
        self._info_print("Using bootstrapping to find initial keypoints")
        bootstrap_frames = self._param_server["initialization"]["bootstrap_frames"]
        if dataset == DataSet.KITTI:
            img0_path = os.path.join(
                self._dataset_paths["KITTI"], '05/image_0', f'{bootstrap_frames[0]:06d}.png')
            img1_path = os.path.join(
                self._dataset_paths["KITTI"], '05/image_0', f'{bootstrap_frames[1]:06d}.png')

            img0 = np.array(cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE))
            img1 = np.array(cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE))
        elif dataset == DataSet.MALAGA:
            if self.left_images is None:
                raise ValueError("No left_images variable available for MALAGA dataset")
            img0_path = os.path.join(
                self._dataset_paths["MALAGA"], 'malaga-urban-dataset-extract-07_rectified_800x600_Images', self.left_images[bootstrap_frames[0]])
            img1_path = os.path.join(
                self._dataset_paths["MALAGA"], 'malaga-urban-dataset-extract-07_rectified_800x600_Images', self.left_images[bootstrap_frames[1]])

            img0 = np.array(cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE))
            img1 = np.array(cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE))
        elif dataset == DataSet.PARKING:
            img0_path = os.path.join(
                self._dataset_paths["PARKING"], 'images', f'img_{bootstrap_frames[0]:05d}.png')
            img1_path = os.path.join(
                self._dataset_paths["PARKING"], 'images', f'img_{bootstrap_frames[1]:05d}.png')

            img0 = np.array(cv2.convertScaleAbs(
                cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)))
            img1 = np.array(cv2.convertScaleAbs(
                cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)))
        else:
            raise AssertionError("Invalid dataset selection")

        self.image_0 = img0

        # setup for continuous operation
        state = self.initialization(img0, img1, self.K, dataset == DataSet.KITTI)
        self.world_pose: NDArray = np.block([[np.eye(3), np.zeros((3, 1))],
                                                [ 0, 0, 0, 1 ]])
        from_index = bootstrap_frames[1] + 1
        to_index = self.last_frame + 1
        prev_img = img1

        return state, range(from_index, to_index), prev_img

        # # do not use bootstrap, initialize with first frame and given keypoints
        # self._info_print("No bootstrapping, using provided keypoints")
        # if dataset != DataSet.KITTI:
        #     raise AssertionError(
        #         "No keypoints provided for initialization for dataset other than KITTI")
        # return self._get_kitti_debug_points()

    def _process_frame(self, curr_image: MatLike, prev_image: MatLike, prev_state: State, frame_id: int, prev_pose: Pose, dataset) -> Tuple[State, Pose]:
        # From the previous image and previous state containing keypoints and landmarks,
        # figure out which keypoints carried over in the new image and return that set of P and X
        updated_state = self.keypoint_tracker(self.K, prev_state, prev_image, curr_image)

        # calling the pose estimator
        pose_success, camera_rot_matrix_wrt_world, camera_trans_vec_wrt_world = self.pose_estimator(updated_state, self.K, prev_pose, frame_id)

        pose = prev_pose
        if pose_success:
            pose = PoseEstimator.cvt_rot_trans_to_pose(camera_rot_matrix_wrt_world, camera_trans_vec_wrt_world)
            updated_state = self.keypoint_tracker.remove_landmarks_behind_the_camera(self.K, updated_state, pose)

            if frame_id % self.keyframe_frequency == 0:
                # Find and triangulate new landmarks
                updated_state = self.landmark_triangulation(self.K, curr_image, prev_image, updated_state, prev_state, pose)

        self._plot_vo_vis_main(dataset, pose, updated_state, curr_image, frame_id)
        return updated_state, pose

    # region Visual Odometry main visualization methods
    def _init_figures(self):
        self.vis_figure, self.vis_axs = plt.subplots(2, 2, figsize=(20, 10))
        self.vis_axs[0, 0].remove()
        self.vis_axs[0, 0] = self.vis_figure.add_subplot(2, 2, 1)
        self.vis_figure.suptitle("Visual Odometry Pipeline")

    def _clear_figures(self):
        for ax in self.vis_axs.flat:
            ax.clear()

    def _refresh_figures(self):
        self.vis_figure.canvas.draw_idle()
        for ax in self.vis_axs.flat:
            ax.legend()
        plt.pause(.1)

    def _plot_full_trajectory(self, fig_id: Tuple[int, int], pose: Pose, frame_id: int = 0):
        """
        Plots the trajectory of the camera wrt the world frame. Plots only the x and z coordinates since the camera
        is moving on a flat plane.
        """
        self.vis_axs[*fig_id].clear()

        # Camera pose wrt world frame
        self.vis_axs[*fig_id].set_title("Full Trajectory")
        for i in range(len(self.poses)):
            last_pose = True if i == len(self.poses)-1 else False
            PlotUtils._plot_trajectory(self.vis_axs[*fig_id], self.poses[i], i, plot_ground_truth=False, ground_truth=self.ground_truth, plot_red=last_pose)

        self.vis_axs[*fig_id].set_xlabel("X position")
        self.vis_axs[*fig_id].set_ylabel("Z position")
        self.vis_axs[*fig_id].set_aspect('equal', adjustable='datalim')

        # if self.dataset == DataSet.PARKING:
        #     self.vis_axs[*fig_id].set_ylim(-10, 10)
        # elif self.dataset == DataSet.KITTI:
        #     self.vis_axs[*fig_id].set_xlim(-10, 10)

    def _plot_trajectory_and_landmarks(self, fig_id: Tuple[int, int], pose: Pose, state: State, frame_id: int = 0):
        """
        Plots the trajectory and the landmarks. Plots only the x and z coordinates since the camera
        is moving on a flat plane.
        """
        # Clearing the axes to show changes in landmarks
        self.vis_axs[*fig_id].clear()

        self.vis_axs[*fig_id].set_title("Trajectory and Landmarks of last 20 frames")

        num_frames_to_plot = min(20, len(self.states))
        for i in range(1, num_frames_to_plot+1):
            PlotUtils._plot_trajectory(self.vis_axs[*fig_id], self.poses[-i], i-1, plot_ground_truth=False, ground_truth=self.ground_truth)
            PlotUtils._plot_landmarks(self.vis_axs[*fig_id], self.poses[-i], self.states[-i], i-1)

        PlotUtils._plot_trajectory(self.vis_axs[*fig_id], self.poses[-1], i-1, plot_ground_truth=False, ground_truth=self.ground_truth, plot_red=True)

        self.vis_axs[*fig_id].set_xlabel("X position")
        self.vis_axs[*fig_id].set_ylabel("Z position")
        self.vis_axs[*fig_id].set_aspect('equal', adjustable='datalim')

    def _plot_keypoint_tracking_count(self, fig_id: Tuple[int, int], state: State, frame_id: int = 0):
        """
        Plots the number of keypoints tracked in each frame.
        """
        # Maintain history of frames and keypoint counts. This is 
        # enable us to plot the history of keypoints tracked as a line
        if not hasattr(self, 'keypoint_history'):
            self.keypoint_history = {'frames': [], 'keypoints': []}

        # Append current frame and keypoint count to the history. 
        self.keypoint_history['frames'].append(frame_id)
        self.keypoint_history['keypoints'].append(state.P.shape[1])

        # Clear the axis for fresh plotting
        self.vis_axs[*fig_id].clear()

        self.vis_axs[*fig_id].set_title("Keypoint Tracking Count")
        self.vis_axs[*fig_id].plot(self.keypoint_history['frames'], self.keypoint_history['keypoints'], marker='o')
        self.vis_axs[*fig_id].set_xlabel("Frame")
        self.vis_axs[*fig_id].set_ylabel("Number of keypoints")
        # self.vis_axs[*fig_id].set_aspect('equal')

    def _plot_trajectory_and_landmarks_history(self, fig_id: Tuple[int, int], pose: Pose, state: State, frame_id: int = 0):
        """
        Continuously plots the landmarks. Plots only the x and z coordinates since the camera
        is moving on a flat plane.
        """
        camera_t_wrt_world = pose[:3, 3]
        landmarks_wrt_world = state.X

        self.vis_axs[*fig_id].set_title("Landmark History")
        if frame_id == 0:
            self.vis_axs[*fig_id].scatter(camera_t_wrt_world[0], camera_t_wrt_world[2], color='red', s=10, label="Pose History")
            self.vis_axs[*fig_id].scatter(landmarks_wrt_world[0, :], landmarks_wrt_world[2, :], color='green', s=10, label="ALL Landmarks")
        else:
            self.vis_axs[*fig_id].scatter(camera_t_wrt_world[0], camera_t_wrt_world[2], color='red', s=10)
            self.vis_axs[*fig_id].scatter(landmarks_wrt_world[0, :], landmarks_wrt_world[2, :], color='green', s=10)

        self.vis_axs[*fig_id].set_xlabel("X position")
        self.vis_axs[*fig_id].set_ylabel("Z position")
        # self.vis_axs[*fig_id].set_aspect('equal')

    def _plot_keypoints_on_frame(self, fig_id: Tuple[int, int], image, state: State, frame_id: int=0):
        """
        Plotting the keypoints on the image
        """
        self.vis_axs[*fig_id].clear()
        self.vis_axs[*fig_id].set_title(f"Keypoints on Frame {frame_id}")

        self.vis_axs[*fig_id].imshow(image, cmap="gray")
        self.vis_axs[*fig_id].scatter(state.P[0, :], state.P[1, :], color="red", s=2)
        self.vis_axs[*fig_id].legend(["Keypoints"])
        # self.vis_axs[*fig_id].set_aspect('equal')

    def _plot_vo_vis_main(self, dataset: DataSet, pose: Pose, state: State, image: np.ndarray, frame_id: int=0):
        """
        Plots all of the subplots in the main visualization.
        """
        self.poses.append(pose)
        self.states.append(state)

        self._plot_keypoints_on_frame((0, 0), image, state, frame_id)
        self._plot_keypoint_tracking_count((1, 0), state, frame_id)
        self._plot_trajectory_and_landmarks((0, 1), pose, state, frame_id)
        self._plot_full_trajectory((1, 1), pose, frame_id)

        # # Makes sure that plots refresh
        # self._refresh_figures()
        # time.sleep(0.01)

        # self.vis_figure.savefig(os.path.join(f'screencasts/{dataset.name}', f'frame_{frame_id:04d}.png'))
    # endregion

    # region RUN
    def run(self, dataset: DataSet = DataSet.KITTI, use_bootstrap: bool = True):
        self._info_print(f"Running pipeline for dataset: {dataset.name}, bootstrap: {use_bootstrap}")

        state, image_range, prev_image = self._init_dataset(dataset, use_bootstrap)
        prev_pose = self.world_pose

        self._plot_vo_vis_main(dataset, self.world_pose, state, self.image_0, frame_id = 0)

        ### Continuous Operation ###
        for frame_id in image_range:
            self._info_print(f"Processing frame {frame_id}")

            match dataset:
                case DataSet.KITTI:
                    image_path = os.path.join(self._dataset_paths["KITTI"], '05/image_0', f'{frame_id:06d}.png')
                    image = np.array(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
                case DataSet.MALAGA:
                    if self.left_images is None:
                        raise ValueError("No left_images variable available for MALAGA dataset")
                    image_path = os.path.join(self._dataset_paths["MALAGA"], 'malaga-urban-dataset-extract-07_rectified_800x600_Images', self.left_images[frame_id])
                    image = np.array(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
                case DataSet.PARKING:
                    image_path = os.path.join(self._dataset_paths["PARKING"], 'images', f'img_{frame_id:05d}.png')
                    image = np.array(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
                    image = cv2.convertScaleAbs(image)

            new_state, new_pose = self._process_frame(image, prev_image, state, frame_id, prev_pose, dataset)

            # Makes sure that plots refresh
            self._refresh_figures()
            time.sleep(0.01)

            # Prepare for the next iteration
            prev_image = image
            state = new_state
            prev_pose = new_pose
    # endregion 


def main():
    args = parse_args() # parse command line arguments

    # load args
    dataset = DataSet[args.dataset]
    debug = LogLevel[args.debug]
    param_server_path = args.params
    use_bootstrap = args.no_bootstrap

    # init param server
    param_server = ParamServer(os.path.join(os.path.dirname(
        os.path.dirname(__file__)), param_server_path))

    plt.ion()

    # init and run pipeline
    pipeline = VisualOdometryPipeline(param_server=param_server, debug=debug)
    pipeline.run(dataset=dataset, use_bootstrap=use_bootstrap)

    plt.ioff()

if __name__ == "__main__":
    main()
