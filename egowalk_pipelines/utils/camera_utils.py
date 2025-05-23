import numpy as np
import cv2

from typing import Tuple


class CameraParameters:

    def __init__(self,
                 fx: float,
                 fy: float,
                 cx: float,
                 cy: float,
                 k1: float,
                 k2: float,
                 k3: float,
                 k4: float,
                 k5: float,
                 k6: float,
                 p1: float,
                 p2: float) -> None:
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._k1 = k1
        self._k2 = k2
        self._k3 = k3
        self._k4 = k4
        self._k5 = k5
        self._k6 = k6
        self._p1 = p1
        self._p2 = p2

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array([[self._fx, 0.0, self._cx],
                         [0.0, self._fy, self._cy],
                         [0.0, 0.0, 1.0]])

    @property
    def distortion_coefficients(self) -> np.ndarray:
        return np.array([self._k1, self._k2, self._p1,
                         self._p2, self._k3, self._k4, self._k5, self._k6])


DEFAULT_CAMERA_PARAMS = CameraParameters(
    fx=367.544,
    fy=367.5575,
    cx=483.335,
    cy=298.59,
    k1=4.19757,
    k2=2.88146,
    k3=0.808998,
    k4=4.20432,
    k5=2.9008,
    k6=1.1033,
    p1=-0.000224475,
    p2=0.000175638
)


def project_points(points: np.ndarray,
                   height: float,
                   camera_params,
                   image_wh: Tuple[int, int]) -> np.ndarray:
    if len(points.shape) == 1:
        single_input = True
        points = points[np.newaxis, :]
    else:
        single_input = False
    if points.shape[1] == 3:
        points = points[:, :2]

    xyz = np.concatenate([points,
                          -height * np.ones(list(points.shape[:-1]) + [1])], axis=-1)

    rvec = tvec = (0, 0, 0)
    xyz_cv = np.stack([-xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
    uv, _ = cv2.projectPoints(
        xyz_cv, rvec, tvec,
        camera_params.camera_matrix,
        camera_params.distortion_coefficients
    )
    pixels = uv[:, 0, :]
    pixels = np.round(pixels).astype(np.int32)

    if image_wh is not None:
        pixels = np.array([p for p in pixels if np.all(p > 0) and np.all(p < [image_wh[0], image_wh[1]])])
        if len(pixels) == 0:
            return None

    if single_input:
        pixels = pixels[0]

    return pixels


def get_depth(depth_img: np.ndarray,
              pixel_coords: tuple[int, int]) -> float:
    x, y = pixel_coords
    depth = depth_img[y, x]
    
    if not np.isnan(depth):
        return depth
    
    # If depth is NaN, find the closest non-NaN pixel
    h, w = depth_img.shape
    min_dist = float('inf')
    closest_depth = None
    
    # Create a mask of valid (non-NaN) depth values
    valid_mask = ~np.isnan(depth_img)
    if not np.any(valid_mask):
        return 0.0  # Return default if no valid depths
    
    # Get coordinates of all valid depth pixels
    valid_coords = np.argwhere(valid_mask)
    
    # Calculate distances to the target pixel
    distances = np.sqrt((valid_coords[:, 0] - y)**2 + (valid_coords[:, 1] - x)**2)
    
    # Find the closest valid pixel
    closest_idx = np.argmin(distances)
    closest_y, closest_x = valid_coords[closest_idx]
    
    return depth_img[closest_y, closest_x]
