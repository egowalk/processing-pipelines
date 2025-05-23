import numpy as np

from pathlib import Path
from PIL import Image
from egowalk_pipelines.models.ram import RAM
from egowalk_pipelines.models.tag2text import Tag2Text
from egowalk_pipelines.models.grounding_dino import GroundingDINOModel
from egowalk_pipelines.models.sam import SAM
from egowalk_pipelines.misc.segments import Segment, BoundingBox
from egowalk_pipelines.misc.types import ImageLike


DEFAULT_TAGS_BLACKLIST = ["room",
                          "building",
                          "floor",
                          "carpet",
                          "wall",
                          "ceiling"]


class RAMGroundingDINOSegmentsExtractor:
    """
    An algorithm for extracting segments from an image using RAM, GroundingDINO and SAM.
    It first uses RAM to generate tags for the image.
    Next, GroundingDINO is used to generate bounding boxes for the tags with non-maximum suppression.
    Finally, SAM is optionally used to generate masks for the bounding boxes.
    Also, if tags blacklist is provided, output segments will be filtered by it.
    """

    def __init__(self,
                 tagging_model: RAM | Tag2Text,
                 grounding_dino: GroundingDINOModel,
                 sam: SAM | None = None,
                 tags_blacklist: list[str] | None = None,
                 stochastic_tags_blacklist: list[str] | None = None):
        """
        Args:
            ram: RAM model.
            grounding_dino: GroundingDINO model.
            sam: SAM model.
            tags_blacklist: List of tags to filter out from the output segments.
            stochastic_tags_blacklist: List of tags to filter out from the output segments with some probability.
        """
        self._tagging_model = tagging_model
        self._grounding_dino = grounding_dino
        self._sam = sam
        self._tags_blacklist = tags_blacklist if tags_blacklist is not None else []
        self._stochastic_tags_blacklist = stochastic_tags_blacklist if stochastic_tags_blacklist is not None else []
    
    def __call__(self,
                 image: ImageLike,
                 return_masks: bool = False) -> list[Segment]:
        """
        Extract segments from an image.

        Args:
            image: Image to extract segments from.
            return_masks: Whether to return masks for the segments.

        Returns:
            List of segments.
        """
        blacklist = self._tags_blacklist
        coin = np.random.rand()
        if coin < 0.5:
            blacklist.extend(self._stochastic_tags_blacklist)
        
        image = self._load_image(image)
        source_tags = self._tagging_model(image)
        source_tags = self._filter_tags(source_tags, blacklist)

        boxes = []
        labels = []
        for source_tag in source_tags:
            boxes_local, labels_local = self._grounding_dino(image, [source_tag])
            boxes.extend(boxes_local)
            labels.extend(labels_local)

        if return_masks and self._sam is not None:
            segments = self._sam(image, boxes)
            segments = [Segment(bbox=segments[i].bbox,
                                crop=segments[i].crop,
                                crop_no_background=segments[i].crop_no_background,
                                mask=segments[i].mask,
                                tags=[labels[i]]) for i in range(len(segments))]
        else:
            segments = [Segment(bbox=box,
                                crop=self._crop_image(image, box),
                                tags=[label]) for box, label in zip(boxes, labels)]
            
        segments = self._filter_segments(segments, blacklist)

        return segments

    def _crop_image(self, image: np.ndarray, box: BoundingBox) -> np.ndarray:
        x, y, w, h = box.xywh
        return image[y:y+h, x:x+w]

    def _filter_tags(self, 
                     tags: list[str],
                     blacklist: list[str]) -> list[str]:
        result = []
        for tag in tags:
            to_add = True
            for blacklist_tag in blacklist:
                if blacklist_tag in tag:
                    to_add = False
                    break
            if to_add:
                result.append(tag)
        return result

    def _filter_segments(self, 
                         segments: list[Segment],
                         blacklist: list[str]) -> list[Segment]:
        result = []
        for segment in segments:
            to_add = True
            for blacklist_tag in blacklist:
                for segment_tag in segment.tags:
                    if blacklist_tag == segment_tag:
                        to_add = False
                        break
            if to_add:
                result.append(segment)
        return result
                        
    def _load_image(self, image: ImageLike) -> np.ndarray:
        if isinstance(image, str) or isinstance(image, Path):
            image = Image.open(image).convert("RGB")
            image = np.array(image)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, np.ndarray):
            image = image.copy()
        else:
            raise ValueError(f"Invalid image type: {type(image)}")
        return image
