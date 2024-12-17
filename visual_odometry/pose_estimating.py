import numpy as np
import matplotlib.pyplot as plt
import cv2

from visual_odometry.common.enums import LogLevel
from visual_odometry.common import BaseClass
from visual_odometry.common import State
from visual_odometry.common import ParamServer

class PoseEstimator(BaseClass):
    def __init__(self, param_server: ParamServer, debug: LogLevel = LogLevel.INFO):
        """
        Pose estimator class. This class is responsible for estimating the camera pose from the tracked keypoints.
        
        :param param_server: ParamServer object containing the parameters.
        :type param_server: ParamServer
        :param debug: Debug level.
        :type debug: LogLevel
        """
        super().__init__(debug)
        self._info_print("Pose estimator initialized.")
        
        # Retrieve required parameters from the ParamServer
        self.params = param_server["pose_estimator"]
        
        self.debug_fig = plt.figure()

    