from visual_odometry.common.enums import LogLevel

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