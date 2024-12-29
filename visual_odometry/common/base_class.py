from visual_odometry.common.enums import LogLevel
from functools import wraps

class BaseClass:
    def __init__(self, debug: LogLevel = LogLevel.INFO):
        self.debug = debug

    def _debug_print(self, msg: str) -> None:
        """Helper method for debug printing."""
        if self.debug >= LogLevel.DEBUG:
            print(f"[{self.__class__.__name__}] {msg}")

    def _info_print(self, msg: str) -> None:
        """Helper method for info printing."""
        if self.debug >= LogLevel.INFO:
            print(f"[{self.__class__.__name__}] {msg}")

    def _debug_visualize(self, *args, **kwargs) -> None:
        """Helper method for debug visualization."""
        if self.debug >= LogLevel.VISUALIZATION:
            self.visualize(*args, **kwargs)

    def visualize(self, *args, **kwargs) -> None:
        raise NotImplementedError("Visualize method not needs to be implemented for the child class.")

    @classmethod
    def plot_debug(cls, func):
        """Decorator implementation of _debug_visualize so that we're not restircted to a single visualize() call"""
        @wraps(func)
        def check(self, *args, **kwargs):
            if self.debug >= LogLevel.VISUALIZATION:
                return func(self, *args, **kwargs)
        return check

    def get_axs(self, *args, **kwargs):
        """Return the axs object so that we can call the generic matplotlib plotting library without worrying about specific functions"""
        if self.debug >= LogLevel.VISUALIZATION:
            return self._get_axs_impl(*args, **kwargs)
        return NoOpAxis()

    def _get_axs_impl(self, *args, **kwargs):
        raise NotImplementedError("Implement the logic to get the figure axis for the corresponding class")

class NoOpAxis:
    """A no-op axis that does nothing when plotting methods are called."""
    def __getattr__(self, name):
        def no_op(*args, **kwargs):
            pass
        return no_op
