try:
    import pyzed.sl as sl
except:
    print("Warning: ZED SDK (pyzed) is not installed, some functions are unavailable")

from typing import Optional
from egowalk_pipelines.misc.types import PathLike


class DurationError(Exception):
    """
    Exception raised for errors during .svo/.svo2 file duration estimation.
    """

    def __init__(self,
                 source_file: PathLike,
                 message: str,
                 cause: Optional[Exception] = None) -> None:
        """
        Initialize the exception with an optional message and cause.

        Args:
            source_file: Path to the .svo/.svo2 file being processed.
            message: A description of the error.
            cause: The original exception that caused this error.
        """
        super(DurationError, self).__init__(message)
        self._source_file = str(source_file)
        self._message = message
        self._cause = cause

    @property
    def source_file(self) -> str:
        return self._source_file

    def __str__(self) -> str:
        """
        Return a string representation of the error.
        """
        if self._cause is not None:
            return f"{self._message} for file {self._source_file} (caused by: {type(self._cause).__name__}: {self._cause})"
        return self._message


def estimate_svo_duration(input_svo_file: PathLike,
                          sdk_verbose: bool = False) -> int:
    """
    Estimate the duration of an .svo/.svo2 file.

    Args:
        input_svo_file: Path to the .svo/.svo2 file being processed.
        sdk_verbose: Whether to enable verbose mode for the ZED SDK.

    Returns:
        The duration of the .svo/.svo2 file in seconds.
    """
    init_params = sl.InitParameters(sdk_verbose=sdk_verbose)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init_params.set_from_svo_file(str(input_svo_file))
    init_params.svo_real_time_mode = False
        
    # Open "camera"
    zed = sl.Camera()
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        msg = f"Failed to open Zed object with error code: {err}"
        zed.close()
        raise DurationError(input_svo_file, msg)
    
    n_frames = zed.get_svo_number_of_frames()
    fps = zed.get_camera_information().camera_configuration.fps
    zed.close()
    
    duration = int(round(n_frames / fps))

    return duration
