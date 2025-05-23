import numpy as np

from dataclasses import dataclass
from egowalk_pipelines.misc.segments import Segment


@dataclass
class SceneObject:
    # ID (index) of the object
    object_id: int

    # Color used to draw the object
    object_color: int

    # Segment of the object
    segment: Segment

    # Center of the object in the BEV (x, y)
    center_bev: np.ndarray

    # Center of the object in the 3D space (x, y, z)
    center_3d: np.ndarray
