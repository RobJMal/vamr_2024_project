# vamr_2024_project
Final project for Vision Algorithms for Mobile Robotics (VAMR) Fall 2024.

## Video Generation and Development
The screencasts were captured on a laptop computer with an i7 3770k processor with 4 cores @ 3.3 GHz and 16 GB RAM. The project was also tested on the following system: 

- OS: Ubuntu 22.04 | CPU: Intel Core I7; 6 cores at @ 2.4 GHz | RAM: 16 GB
- OS: macOS Sonoma 14.1.1 | CPU: Intel Core I5 (8th gen); 4 cores at @ 2.3 GHz | RAM: 8 GB

It should be noted that the pipeline is single-threaded. 

## Setup
The setup process for our repository is separated into two steps: 
- Environment setup
- Dataset download and setup

### Environment setup 
To create a conda environment with all of the dependencies and Python version, run the following commands. Because the visual odometry pipeline we created is a Python package, it will need to be installed as one, hence the second commands:
```
conda env create -f environment.yml
pip install -e .
```

Alternatively, install the package dependencies direclty using pip install. Note that your Python version should be 3.12 or greater:
```
pip install -r requirements.txt
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

