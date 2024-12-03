import numpy as np
import os
import glob
import cv2
import time

from visual_odometry.common import ParamServer
from visual_odometry.common import BaseClass
from visual_odometry.common.enums import LogLevel
from visual_odometry.common.enums import DataSet
from visual_odometry.initialization import Initialization


class VisualOdometryPipeline(BaseClass):
    def __init__(self, param_server: ParamServer, debug: LogLevel = LogLevel.INFO):
        super().__init__(debug)
        self._param_server = param_server

        self._dataset_paths = {"KITTI": os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets/kitti"),
                               "MALAGA": os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets/malaga-urban-dataset-extract-07"),
                               "PARKING": os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets/parking")}

        self._info_print(
            f"VO monocular pipeline initialized\n - Log level: {debug.name}\n - ParamServer: {self._param_server}")

    def run(self, dataset: DataSet = DataSet.KITTI):
        self._info_print(f"Running pipeline for dataset: {dataset.name}")
        # Setup
        if dataset == DataSet.KITTI:
            ground_truth = np.loadtxt(
                self._dataset_paths["KITTI"] + '/poses/05.txt')
            ground_truth = ground_truth[:, [-9, -1]]
            last_frame = 2760
            K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                         [0, 7.188560000000e+02, 1.852157000000e+02],
                         [0, 0, 1]])
        elif dataset == DataSet.MALAGA:
            images_path = os.path.join(
                self._dataset_paths["MALAGA"], 'malaga-urban-dataset-extract-07_rectified_800x600_Images')
            images = sorted(glob.glob(os.path.join(images_path, '*')))

            left_images = images[2::2]
            last_frame = len(left_images) - 1

            K = np.array([[621.18428, 0, 404.0076],
                          [0, 621.18428, 309.05989],
                          [0, 0, 1]])

        elif dataset == DataSet.PARKING:
            last_frame = 598
            K = np.loadtxt(
                self._dataset_paths["PARKING"] + '/K.txt', delimiter=',')
            ground_truth = np.loadtxt(
                self._dataset_paths["PARKING"] + '/poses.txt')
            ground_truth = ground_truth[:, [-9, -1]]
        else:
            raise AssertionError("Invalid dataset selection")

        #  Bootstrap
        bootstrap_frames = self._param_server["initialization"]["bootstrap_frames"]
        if dataset == DataSet.KITTI:
            img0_path = os.path.join(
                self._dataset_paths["KITTI"], '05/image_0', f'{bootstrap_frames[0]:06d}.png')
            img1_path = os.path.join(
                self._dataset_paths["KITTI"], '05/image_0', f'{bootstrap_frames[1]:06d}.png')

            img0 = np.array(cv2.imread(img0_path))
            img1 = np.array(cv2.imread(img1_path))
        elif dataset == DataSet.MALAGA:
            img0_path = os.path.join(
                self._dataset_paths["MALAGA"], 'malaga-urban-dataset-extract-07_rectified_800x600_Images', left_images[bootstrap_frames[0]])
            img1_path = os.path.join(
                self._dataset_paths["MALAGA"], 'malaga-urban-dataset-extract-07_rectified_800x600_Images', left_images[bootstrap_frames[1]])

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

        # TODO: implement initialization
        initialization = Initialization(
            self._param_server, debug=LogLevel.INFO)
        state = initialization(img0, img1)

        # Continuous Operation
        """
        for i in range(bootstrap_frames[1] + 1, last_frame + 1):
            self._info_print(f"\n\nProcessing frame {i}\n=====================")
            if dataset == DataSet.KITTI:
                image_path = os.path.join(self._dataset_paths["KITTI"], '05/image_0', f'{i:06d}.png')
                image = np.array(cv2.imread(image_path))
            elif dataset == DataSet.MALAGA:
                image_path = os.path.join(self._dataset_paths["MALAGA"], 'malaga-urban-dataset-extract-07_rectified_800x600_Images', left_images[i])
                image = np.array(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
            elif dataset == DataSet.PARKING:
                image_path = os.path.join(self._dataset_paths["PARKING"], 'images', f'img_{i:05d}.png')
                image = np.array(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
                image = cv2.convertScaleAbs(image)
            else:
                raise AssertionError("Invalid dataset selection")

        # TODO: implement continuous operation on "image"
        
        # Makes sure that plots refresh
        time.sleep(0.01)
        
        prev_img = image
        """


def main():
    param_server = ParamServer(os.path.join(os.path.dirname(
        os.path.dirname(__file__)), "params/pipeline_params.yaml"))
    pipeline = VisualOdometryPipeline(param_server=param_server)
    pipeline.run()


if __name__ == "__main__":
    main()
