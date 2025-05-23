from __future__ import annotations

import numpy as np

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class BoundingBox:
    """
    A bounding box of the object in the image.
    Default format is (x, y, w, h); other formats produced from this format.
    """

    xywh: Tuple[int, int, int, int]
    """
    (x, y) is the top-left corner of the bounding box.
    (w, h) is the width and height of the bounding box.
    """
    
    xyxy: Tuple[int, int, int, int] = field(init=False)
    """
    (x1, y1) is the top-left corner of the bounding box.
    (x2, y2) is the bottom-right corner of the bounding box.
    """

    center: Tuple[int, int] = field(init=False)
    """
    Center of the bounding box.
    """

    area: int = field(init=False)
    """
    Area of the bounding box.
    """

    @staticmethod
    def from_xywh(xywh: Tuple[int, int, int, int]) -> BoundingBox:
        """
        Create a BoundingBox from the (x, y, w, h) format.
        """
        return BoundingBox(xywh)

    @staticmethod
    def from_xyxy(xyxy: Tuple[int, int, int, int]) -> BoundingBox:
        """
        Create a BoundingBox from the (x1, y1, x2, y2) format.
        """
        x1, y1, x2, y2 = xyxy
        return BoundingBox((x1, y1, x2 - x1, y2 - y1))

    def __post_init__(self) -> None:
        x, y, w, h = self.xywh
        if x < 0 or y < 0 or w < 0 or h < 0:
            raise ValueError("Bounding box dimensions must be non-negative")
        self.xyxy = (x, y, x + w, y + h)
        self.center = (x + w // 2, y + h // 2)
        self.area = w * h
        
    def __str__(self) -> str:
        return f"BoundingBox(xywh={self.xywh})"

    def __repr__(self) -> str:
        return f"{{'xywh': {self.xywh}}}"


@dataclass
class Segment:
    """
    Segment (in the RoboHop notation) of the object in the image.
    Can be result of extraction from models like SAM, RAM+Grounded-SAM, etc.
    """
    
    bbox: BoundingBox | None = None
    """
    Bounding box of the segment in the image.
    """
    mask: np.ndarray | None = None
    """
    Mask of the segment in the image.
    """
    crop: np.ndarray | None = None
    """
    Crop of the segment in the image.
    """
    crop_no_background: np.ndarray | None = None
    """
    Crop of the segment in the image without background.
    """
    tags: list[str] | None = None
    """
    Tags of the segment.
    """

    def __str__(self) -> str:
        return f"Segment(bbox={self.bbox})"

    def __repr__(self) -> str:
        fields = []
        if self.bbox is not None:
            fields.append(f"'bbox': {self.bbox}")
        if self.mask is not None:
            fields.append(f"'mask': array(shape={self.mask.shape})")
        if self.crop is not None:
            fields.append(f"'crop': array(shape={self.crop.shape})")
        if self.crop_no_background is not None:
            fields.append(f"'crop_no_background': array(shape={self.crop_no_background.shape})")
        if self.tags:
            fields.append(f"'tags': {self.tags}")
        return f"{{{', '.join(fields)}}}"
