# vamr_2024_project
Final project for Vision Algorithms for Mobile Robotics (VAMR) Fall 2024.

## Setup
To install the necessary dependencies, run the following command:
```
pip install -e .
```

## Running the pipeline
To run the VO pipeline, you would run the following:
```
python visual_odometry/main.py
```

The following are the command line arguments that could be passed in:
-  `--dataset`
    - Dataset to run the pipeline on
    - Values: `KITTI`, `PARKING`, `MALAGA`
    - Type: string
    - Default: `KITTI`
-  `--debug`
    - Debug level
    - Values: `NONE`, `INFO`, `DEBUG`, `VISUALIZATION`
    - Type: string
    - Default: `INFO`
-  `--params`
    - Path to parameters file
    - Values: `KITTI`, `PARKING`, `MALAGA`
    - Type: string
    - Default: `params/pipeline_params.yaml`
-  `--no-bootstrap`
    - Do not use bootstrap, initialize based on given keypoints
    - Default: `store_false`

For example, to run the `PARKING` dataset with `INFO`-level debugging:
```
python visual_odometry/main.py --dataset PARKING --debug INFO
```
