import enum


EXTENSION_SVO = "svo"
"""ZED SDK SVO v1 extension."""
EXTENSION_SVO2 = "svo2"
"""ZED SDK SVO v2 extension."""

METADATA_FILE_NAME = "metadata.json"
"""Metadata file name in the EgoWalk recordings."""

DEFAULT_FPS = 30
"""Default FPS of the raw EgoWalk recordings."""


VIEW_LEFT = "left"
"""Left view."""
VIEW_RIGHT = "right"
"""Right view."""
VIEW_BOTH = "both"
"""Both views."""


class DepthMode(enum.Enum):
    """
    Depth extraction mode for ZED SDK.

    See https://www.stereolabs.com/docs/depth-sensing/depth-settings
    """
    PERFORMANCE = "performance"
    QUALITY = "quality"
    ULTRA = "ultra"
    NEURAL = "neural"


DATA_DIR = "data"
"""Data directory."""
VIDEO_DIR = "video"
"""Video directory."""
RGB_DIR = "rgb"
"""RGB directory."""
DEPTH_DIR = "depth"
"""Depth directory."""
ODOMETRY_DIR = "odometry"
"""Odometry directory."""
