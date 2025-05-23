import numpy as np

from typing import Any
from egowalk_pipelines.models.segments_extraction import RAMGroundingDINOSegmentsExtractor
from egowalk_pipelines.utils.camera_utils import CameraParameters, DEFAULT_CAMERA_PARAMS, get_depth
from egowalk_pipelines.annotation.objects import SceneObject
from egowalk_pipelines.annotation.filters import AbstractObjectsFilter, AbstractObjectSelector


class SceneObjectFinder:

    def __init__(self,
                 segments_extractor: RAMGroundingDINOSegmentsExtractor,
                 camera_params: CameraParameters | None = None):
        self._segments_extractor = segments_extractor
        if camera_params is None:
            camera_params = DEFAULT_CAMERA_PARAMS
        self._K_inv = np.linalg.inv(camera_params.camera_matrix)

    def __call__(self,
                 rgb_image: np.ndarray,
                 depth_image: np.ndarray) -> list[SceneObject]:
        segments = self._segments_extractor(rgb_image,
                                  return_masks=False)
        result = []
        for i, segment in enumerate(segments):
            bbox_center = segment.bbox.center
            pixel_depth = get_depth(depth_image, (bbox_center[0], bbox_center[1]))
            bbox_center_world = (self._K_inv @ np.array([bbox_center[0], bbox_center[1], 1.])) * pixel_depth
            bbox_center_world = np.array([bbox_center_world[2], 
                                        -bbox_center_world[0], 
                                        -bbox_center_world[1]])
            result.append(SceneObject(object_id=i,
                                    object_color=tuple(np.random.randint(0, 255, size=3).tolist()),
                                    segment=segment,
                                    center_bev=bbox_center_world[:2].copy(),
                                    center_3d=bbox_center_world.copy()))
        return result


class SceneObjectSelector:

    def __init__(self,
                 filters: list[AbstractObjectsFilter],
                 selector: AbstractObjectSelector):
        self._filters = filters
        self._selector = selector

    def __call__(self,
                 objects: list[SceneObject],
                 context: dict[str, Any]) -> list[SceneObject]:
        for filter in self._filters:
            objects = filter(objects, context)
        return self._selector(objects, context)
