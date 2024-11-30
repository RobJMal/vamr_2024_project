from visual_odometry.common.enums import LogLevel

class BaseClass:
    def __init__(self, debug: LogLevel = LogLevel.TEXT):
        self.debug = debug

    def _debug_print(self, msg: str) -> None:
        """Helper method for debug printing."""
        if self.debug >= LogLevel.TEXT:
            print(f"[{self.__class__.__name__}] {msg}")

    def _debug_visuaize(self) -> None:
        """Helper method for debug visualization."""
        if self.debug >= LogLevel.VISUALIZATION:
            self.visualize()

    def visualize(self, *args, **kwargs) -> None:
        raise NotImplementedError("Visualize method not needs to be implemented for the child class.")