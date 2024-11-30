import numpy as np


class VisualOdometryPipeline:

    def run(self, ds: int = 0):
        print("Driver Method for Visual Odometry Pipeline")
        kitti_path = '../datasets/kitti'
        malaga_path = '../datasets/malaga-urban-dataset-extract-07'
        parking_path = '../datasets/parking'

        # Setup
        if ds == 0:
            print("Using KITTI dataset")
            ground_truth = np.loadtxt(kitti_path + '/poses/05.txt')
            ground_truth = ground_truth[:, [-9, -1]]
            last_frame = 4540
            K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                         [0, 7.188560000000e+02, 1.852157000000e+02],
                         [0, 0, 1]])
        elif ds == 1:
            pass
        elif ds == 2:
            pass
        else:
            assert (False)

        pass


def main():
    pipeline = VisualOdometryPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
