import numpy as np

try:
    import pyzed.sl as sl
except:
    print("Warning: ZED SDK (pyzed) is not installed, some functionality is unavailable")

from typing import Any, Optional, Dict, Tuple, Union
from pathlib import Path
from egowalk_pipelines.misc.types import PathLike, RotTrans
from egowalk_pipelines.extraction.channels import ExtractionChannelError
from egowalk_pipelines.misc.constants import VIEW_LEFT, VIEW_RIGHT, VIEW_BOTH, DepthMode


class DummyChannelReader:
    """
    Dummy channel reader that does nothing.

    Follows the protocol of the ExtractionChannelReader.
    """

    def prepare_params(self,
                       init_params) -> None:
        pass

    def setup(self,
              raw_recording_file_path: PathLike,
              zed) -> None:
        pass

    def read(self,
             zed) -> Optional[Any]:
        pass

    def close(self):
        pass


class RGBChannelReader:
    """
    Image channel reader. Uses ZED SDK to obtain left and right images.

    At each grab step, returns a pair of images (left, right).
    If the view is "left", the left image is returned.
    If the view is "right", the right image is returned.
    If the view is "both", both images are returned.

    Follows the protocol of the ExtractionChannelReader.
    """

    def __init__(self, view: str):
        """
        Initialize the image channel reader.

        Args:
            view: One of the possible views: "left", "right" or "both".
        """
        possible_views = (VIEW_LEFT,
                          VIEW_RIGHT,
                          VIEW_BOTH)
        assert view in possible_views, f"Wrong view {view}, possible values are {possible_views}"
        self._view = view
        self._image_left = None  # sl.Mat
        self._image_right = None  # sl.Mat

    def prepare_params(self,
                       init_params) -> None:
        pass

    def setup(self,
              raw_recording_file_path: PathLike,
              zed) -> None:
        if self._view in (VIEW_LEFT,
                          VIEW_BOTH):
            self._image_left = sl.Mat()
        if self._view in (VIEW_RIGHT,
                          VIEW_BOTH):
            self._image_right = sl.Mat()

    def read(self, zed) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._view == VIEW_LEFT:
            zed.retrieve_image(self._image_left, sl.VIEW.LEFT)
            return self._image_left.get_data(), None
        if self._view == VIEW_RIGHT:
            zed.retrieve_image(self._image_right, sl.VIEW.RIGHT)
            return None, self._image_right.get_data()
        zed.retrieve_image(self._image_left, sl.VIEW.LEFT)
        zed.retrieve_image(self._image_right, sl.VIEW.RIGHT)
        return self._image_left.get_data(), self._image_right.get_data()

    def close(self) -> None:
        self._image_left = None
        self._image_right = None


class OdometryChannelReader:

    _KEY_SDK_VERSION = "extraction_zed_sdk_version"
    _KEY_TRACKER_MODE = "extraction_zed_tracker_mode"

    def __init__(self,
                 gen_2_enabled: bool = True,
                 verbose: bool = False):
        self._gen_2_enabled = gen_2_enabled
        self._verbose = verbose
        self._tracker_mode = sl.POSITIONAL_TRACKING_MODE.GEN_2 \
            if gen_2_enabled else sl.POSITIONAL_TRACKING_MODE.GEN_1
        self._pose = None  # sl.Pose
        self._translation = None  # sl.Translation
        self._tracking_params = None  # sl.PositionalTrackingParameters
        self._sdk_version = None  # str

    def prepare_params(self,
                       init_params) -> None:
        pass

    def setup(self,
              raw_recording_file_path: PathLike,
              zed) -> None:
        # Odometry initialization
        tracking_params = sl.PositionalTrackingParameters()
        tracking_params.enable_imu_fusion = True
        tracking_params.mode = self._tracker_mode
        err = zed.enable_positional_tracking(tracking_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise ExtractionChannelError(
                f"Failed to enable positional tracking, error code: {err}")
        self._tracking_params = tracking_params

        self._pose = sl.Pose()
        self._translation = sl.Translation()

        # Save SDK version used for odometry just in case
        self._sdk_version = zed.get_sdk_version()

    def read(self, zed) -> Optional[RotTrans]:
        # Call ZED odometry API
        tracking_state = zed.get_position(self._pose, sl.REFERENCE_FRAME.WORLD)

        if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
            # Everything is ok, get rotation (quaternion) and translation (vector)
            rotation = self._pose.get_orientation().get()
            translation = self._pose.get_translation(self._translation).get()
            rotation = (rotation[0], rotation[1],
                        rotation[2], rotation[3])
            translation = (translation[0],
                           translation[1],
                           translation[2])
            pose = rotation + translation
            return pose

        if tracking_state == sl.POSITIONAL_TRACKING_STATE.UNAVAILABLE:
            # Unavailable state, as we observed, may be result of the bad camera movements.
            # In other words, odometry couldn't estimate the pose.
            # Unlike in the "off" state, after some iterations odometry can recover automatically.
            # So, we just return None here.
            self._print("Tracking state unavailable")
            return None

        if tracking_state == sl.POSITIONAL_TRACKING_STATE.OFF:
            # Tracking may become off in some cases.
            # We think that this happens due to the "bad" odometry.
            # Unlike in "unavailable" state, odometry just "disables" forever.
            # So, in this case we try to re-enable it with the same parameters.
            # In case of success, odometry will be re-enabled and start in zero position.
            self._print("Tracking is off, re-enabling...")
            err = zed.enable_positional_tracking(self._tracking_params)
            if err != sl.ERROR_CODE.SUCCESS:
                raise ExtractionChannelError(
                    "Failed to re-enable positional tracking")
            self._print("Tracking re-enabled")
            # Return None since odometry will give pose only at next iteration
            return None

        raise ExtractionChannelError(
            f"Unknown return tracking state {tracking_state}")

    def close(self):
        self._pose = None
        self._translation = None
        self._tracking_params = None
        self._sdk_version = None

    def _print(self, msg: str):
        if self._verbose:
            print(msg)


class DepthChannelReader:

    def __init__(self,
                 mode: Union[str, DepthMode] = DepthMode.NEURAL):
        if isinstance(mode, str):
            mode = DepthMode(mode)
        self._mode = mode
        self._image = None  # sl.Mat

    def prepare_params(self,
                       init_params) -> None:
        if self._mode == DepthMode.PERFORMANCE:
            init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        elif self._mode == DepthMode.QUALITY:
            init_params.depth_mode = sl.DEPTH_MODE.QUALITY
        elif self._mode == DepthMode.ULTRA:
            init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        elif self._mode == DepthMode.NEURAL:
            init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        else:
            raise ValueError(f"Unsupported depth mode: {self._mode}")

    def setup(self,
              raw_recording_file_path: PathLike,
              zed) -> None:
        self._image = sl.Mat()

    def read(self, zed) -> np.ndarray:
        zed.retrieve_measure(self._image, sl.MEASURE.DEPTH)
        return self._image.get_data()

    def close(self) -> None:
        self._image = None
