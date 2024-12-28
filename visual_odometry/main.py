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
from visual_odometry.initialization import Initialization
from visual_odometry.keypoint_tracking import KeypointTracker
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

        self.left_images: Optional[list[str]] = None # Will be None if dataset is not Malaga

        self.keypoint_tracker = KeypointTracker(param_server=self._param_server, debug=self.debug)
        self.pose_estimator = PoseEstimator(param_server=self._param_server, debug=self.debug)

        self._info_print(
            f"VO monocular pipeline initialized\n - Log level: {debug.name}\n - ParamServer: {self._param_server}")

    def _get_dataset_metadata(self, dataset: DataSet) -> None:
        # Setup
        match dataset:
            case DataSet.KITTI:
                ground_truth = np.loadtxt(
                    self._dataset_paths["KITTI"] + '/poses/05.txt')
                ground_truth = ground_truth[:, [-9, -1]]
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
            case DataSet.PARKING:
                self.last_frame: int = 598
                self.K: NDArray = np.loadtxt(
                    self._dataset_paths["PARKING"] + '/K.txt', delimiter=',')
                ground_truth = np.loadtxt(
                    self._dataset_paths["PARKING"] + '/poses.txt')
                ground_truth = ground_truth[:, [-9, -1]]

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

        #  Bootstrap
        if use_bootstrap:
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

            initialization = Initialization(param_server=self._param_server, debug=self.debug)

            # setup for continuous operation
            state = initialization(img0, img1, self.K, dataset == DataSet.KITTI)
            from_index = bootstrap_frames[1] + 1
            to_index = self.last_frame + 1
            prev_img = img1

            return state, range(from_index, to_index), prev_img

        # do not use bootstrap, initialize with first frame and given keypoints
        self._info_print("No bootstrapping, using provided keypoints")
        if dataset != DataSet.KITTI:
            raise AssertionError(
                "No keypoints provided for initialization for dataset other than KITTI")
        return self._get_kitti_debug_points()

    def _process_frame(self, curr_image: MatLike, prev_image: MatLike, prev_state: State) -> Tuple[State, NDArray]:
        # From the previous image and previous state containing keypoints and landmarks,
        # figure out which keypoints carried over in the new image and return that set of P and X
        state = self.keypoint_tracker(prev_state, prev_image, curr_image)

        # calling the pose estimator
        pose = PoseEstimator.cvt_rot_trans_to_pose(*self.pose_estimator(state, self.K))

        return state, pose

    def run(self, dataset: DataSet = DataSet.KITTI, use_bootstrap: bool = True):
        self._info_print(f"Running pipeline for dataset: {dataset.name}, bootstrap: {use_bootstrap}")

        state, image_range, prev_image = self._init_dataset(dataset, use_bootstrap)

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

            new_state, pose = self._process_frame(image, prev_image, state)

            # Makes sure that plots refresh
            time.sleep(0.01)

            # Prepare for the next iteration
            prev_image = image
            state = new_state


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

    # init and run pipeline
    pipeline = VisualOdometryPipeline(param_server=param_server, debug=debug)
    pipeline.run(dataset=dataset, use_bootstrap=use_bootstrap)


if __name__ == "__main__":
    main()
