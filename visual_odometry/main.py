import numpy as np
import os

from visual_odometry.common import ParamServer
from visual_odometry.common import BaseClass
from visual_odometry.common.enums import LogLevel
from visual_odometry.common.enums import DataSet


class VisualOdometryPipeline(BaseClass):
    def __init__(self, param_server: ParamServer, debug: LogLevel = LogLevel.INFO):
        super().__init__(debug)
        self._param_server = param_server

        self._dataset_paths = {"KITTI": os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets/kitti"),
                               "MALAGA": os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets/malaga-urban-dataset-extract-07"),
                               "PARKING": os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets/parking")}

        self._info_print(f"Keypoint tracker initialized\n  -Log level: {debug.name}\n - ParamServer: {self._param_server}")
        



    def run(self, dataset: DataSet = DataSet.KITTI):
        self._info_print(f"Running pipeline for dataset: {dataset.name}")
        # Setup
        if dataset == DataSet.KITTI:
            ground_truth = np.loadtxt(self._dataset_paths["KITTI"] + '/poses/05.txt')
            ground_truth = ground_truth[:, [-9, -1]]
            last_frame = 4540
            K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                         [0, 7.188560000000e+02, 1.852157000000e+02],
                         [0, 0, 1]])
        elif dataset == 1:
            pass
        elif dataset == 2:
            pass
        else:
            assert (False)

        pass


def main():
    param_server = ParamServer(os.path.join(os.path.dirname(os.path.dirname(__file__)), "params/pipeline_params.yaml"))
    pipeline = VisualOdometryPipeline(param_server=param_server)
    pipeline.run()


if __name__ == "__main__":
    main()
