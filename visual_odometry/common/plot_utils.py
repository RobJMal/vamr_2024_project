from matplotlib.axes import Axes

from visual_odometry.common.state import Pose


class PlotUtils:
    """
    Static methods for plotting given a plt axs object
    """

    @staticmethod
    def _plot_pose(axs: Axes, pose: Pose, isWorld: bool):
        """
        Pose is always extrinsic (camera frame wrt world frame) so we can directly plot is.
        """
        # Camera position (origin of the camera frame)
        scale = 1
        R = pose[:3, :3]
        t = pose[:3, 3]

        x_axis = (R[:, 0] - t) * scale
        y_axis = (R[:, 1] - t) * scale
        z_axis = (R[:, 2] - t) * scale

        # Plot the camera position as a red dot
        axs.scatter(t[0], t[1], t[2], color='black' if isWorld else 'brown', s=10)

        # Plot the camera axes (X, Y, Z axes) using the rotation matrix
        axs.quiver(t[0], t[1], t[2], x_axis[0], x_axis[1], x_axis[2], color='r', length=scale)
        axs.quiver(t[0], t[1], t[2], y_axis[0], y_axis[1], y_axis[2], color='g', length=scale)
        axs.quiver(t[0], t[1], t[2], z_axis[0], z_axis[1], z_axis[2], color='b', length=scale)
