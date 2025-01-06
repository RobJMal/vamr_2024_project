# Monocular Vision Odometry (VO) Pipeline
Final mini-project for ETH/UZH course Vision Algorithms for Mobile Robotics (VAMR) Fall 2024.

## Setup
The setup process for our repository is separated into two steps: 
- Environment setup
- Dataset download and setup

### Environment setup 
To create a conda environment with all of the dependencies and Python version, run the following commands. Because the visual odometry pipeline we created is a Python package, it will need to be installed as one, hence the second commands:
```bash
conda env create -f environment.yml
pip install -e .
```

Alternatively, install the package dependencies direclty using pip install. Note that your Python version should be 3.12 or greater:
```bash
pip install -r requirements.txt
pip install -e .
```

### Dataset download and setup 
The datasets were obtained from the class website ([click here and navigate to Optional Mini Project section](https://rpg.ifi.uzh.ch/teaching.html)). Because they are zip files, we will need to extract them and make sure that they following the structure outlined below in the `datasets/` directory: 

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
│   │   └── kp_for_debug.txt
|   ├── malaga-urban-dataset-extract-07/
|   |   ├── ...
│   ├── parking/
│   │   ├── images/
│   │   ├── K.txt
│   │   └── poses.txt
├── .gitignore
├── params/
├── visual_odometry/
├── environment.yml
├── README.md
├── requirements.txt
└── setup.py
```

## Running the pipeline
To run the VO pipeline, you would run one of the following commands:
```bash
python visual_odometry/main.py --dataset PARKING    # Runs pipeline with PARKING dataset
python visual_odometry/main.py --dataset KITTI      # Runs pipeline with KITTI dataset
python visual_odometry/main.py --dataset MALAGA     # Runs pipeline with MALAGA dataset
```

To help in the development process, we also added several command-line arguments. They are listed below along with their respective descriptions and can be added on to the commands: 
-  `--dataset`
    - Description: Dataset to run the pipeline on
    - Values: `KITTI`, `PARKING`, `MALAGA`
    - Type: string
    - Default: `KITTI`
-  `--debug`
    - Description: Debug level
    - Values: `NONE`, `INFO`, `DEBUG`, `VISUALIZATION`
    - Type: string
    - Default: `INFO`
-  `--params`
    - Description: Path to parameters file
    - Values: `KITTI`, `PARKING`, `MALAGA`
    - Type: string
    - Default: `params/pipeline_params.yaml`
-  `--no-bootstrap`
    - Description: Do not use bootstrap, initialize based on given keypoints
    - Default: `store_false`

## Video Generation and Development
The screencasts were captured on a laptop computer with an i7 3770k processor with 4 cores @ 3.3 GHz and 16 GB RAM. The project was also tested on the following system: 

- OS: Ubuntu 22.04          | CPU: Intel Core I7; 6 cores at @ 2.4 GHz              | RAM: 16 GB
- OS: macOS Sonoma 14.1.1   | CPU: Intel Core I5 (8th gen); 4 cores at @ 2.3 GHz    | RAM: 8 GB
- OS: macOS Sequoia 15      | CPU: Apple M2 Chip; 8 cores @ 2.42 GHz                | RAM: 16 GB

It should be noted that the pipeline is single-threaded. 

