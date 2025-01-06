# vamr_2024_project
Final project for Vision Algorithms for Mobile Robotics (VAMR) Fall 2024.

## Setup
The setup process for our repository is separated into two steps: 
- Environment setup
- Dataset download and setup

### Environment setup 
To create a conda environment with all of the dependencies and Python version, run the following command:
```
conda env create -f environment.yml
```

If you are not running Conda, run the following command. Ensure that your Python's environment is 3.12 or greater:
```
pip install -e .
```

### Dataset download and setup 
The datasets were obtained from the class website. Because they are zip files, we will need to extract them and make sure that they following the structure outlined below in the `datasets/` directory: 

```
VAMR_2024_PROJECT/
├── .vscode/
├── datasets/
│   ├── kitti/
│   │   ├── 05/
│   │   │   ├── image_0/
│   │   │   ├── image_1/
│   │   │   ├── calib.txt
│   │   │   ├── times.txt
│   │   │   └── poses/
│   │   ├── kp_for_debug.txt
│   │   └── malaga-urban-dataset-extract-07/
│   ├── parking/
│   │   ├── images/
│   │   ├── K.txt
│   │   └── poses.txt
├── .gitignore
├── kitti05.zip
├── malaga-urban-dataset-extract-07.zip
├── params/
├── visual_odometry/
├── visual_odometry.egg-info/
├── environment.yml
├── README.md
├── requirements.txt
└── setup.py
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

