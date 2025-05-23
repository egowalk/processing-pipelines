import numpy as np

from abc import ABC, abstractmethod
from typing import Any
from scipy.spatial.distance import cdist
from canguro_processing_tools.annotation.objects import SceneObject


class AbstractObjectsFilter(ABC):

    @abstractmethod
    def __call__(self, 
               objects: list[SceneObject],
               context: dict[str, Any]) -> list[SceneObject]:
        pass


class AbstractObjectSelector(ABC):

    @abstractmethod
    def __call__(self, 
               objects: list[SceneObject],
               context: dict[str, Any]) -> SceneObject | None:
        pass


class AreaObjectsFilter(AbstractObjectsFilter):

    def __init__(self,
                 threshold_percent: float):
        super(AreaObjectsFilter, self).__init__()
        self._threshold_percent = threshold_percent

    def __call__(self, 
               objects: list[SceneObject],
               context: dict[str, Any]) -> list[SceneObject]:
        img_wh = context["image_wh"]
        img_area = img_wh[0] * img_wh[1]
        result = [e for e in objects if e.segment.bbox.area < self._threshold_percent * img_area]
        return result
    

class CloseObjectsFilter(AbstractObjectsFilter):

    def __init__(self,
                 min_distance: float):
        super(CloseObjectsFilter, self).__init__()
        self._min_distance = min_distance

    def __call__(self, 
               objects: list[SceneObject],
               context: dict[str, Any]) -> list[SceneObject]:
        result = [e for e in objects if np.linalg.norm(e.center_bev) >= self._min_distance]
        return result


class FarTrajObjectsFilter(AbstractObjectsFilter):

    def __init__(self,
                 max_distance: float):
        super(FarTrajObjectsFilter, self).__init__()
        self._max_distance = max_distance

    def __call__(self, 
               objects: list[SceneObject],
               context: dict[str, Any]) -> list[SceneObject]:
        if len(objects) == 0:
            return []
        centers_bev = np.array([e.center_bev for e in objects])
        trajectory_bev = context["trajectory_bev"][:, :2]
        distances = cdist(centers_bev, trajectory_bev)
        min_distances = np.min(distances, axis=1)
        min_distances_dict = {e.object_id: min_distances[i] for i, e in enumerate(objects)}
        result = [e for i, e in enumerate(objects) if min_distances[i] <= self._max_distance]
        min_distances_dict = {e.object_id: min_distances_dict[e.object_id] for e in result}
        context["bev_distances"] = min_distances_dict
        return result


class ClosestTrajObjectSelector(AbstractObjectSelector):

    def __init__(self):
        super(ClosestTrajObjectSelector, self).__init__()

    def __call__(self, 
               objects: list[SceneObject],
               context: dict[str, Any]) -> SceneObject | None:
        if len(objects) == 0:
            return None
        min_distances_dict = context["bev_distances"]
        if len(min_distances_dict) == 0:
            return None
        min_distance = np.inf
        result = None
        for object in objects:
            obj_distance = min_distances_dict[object.object_id]
            if obj_distance < min_distance:
                min_distance = obj_distance
                result = object
        return result
