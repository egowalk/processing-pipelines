import enum
import cv2
import torch
import numpy as np

from pathlib import Path
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from PIL import Image
from egowalk_pipelines.misc.types import PathLike, ImageLike
from egowalk_pipelines.misc.segments import Segment, BoundingBox


class SAMModelType(enum.Enum):
    """
    Enum for the different SAM model types.
    """
    VIT_H = "vit_h"
    VIT_L = "vit_l"
    VIT_B = "vit_b"


class MaskCriterion(enum.Enum):
    SCORE = "score"
    AREA = "area"
    ALL = "all"


class SAM:
    """
    Wrapper over the Segment Anything Model's automatic mask generator.
    """

    def __init__(self,
                 model_type: SAMModelType,
                 checkpoint_path: PathLike,
                 device: str = "cuda"):
        """
        Initialize the SAM model.

        Args:
            model_type: The type of SAM model to use.
            checkpoint_path: The path to the SAM checkpoint.
            device: The device to run the model on.
        """
        if isinstance(model_type, str):
            model_type = SAMModelType(model_type)

        self._model = sam_model_registry[model_type.value](checkpoint=str(checkpoint_path)).to(device)
        self._mask_generator = SamAutomaticMaskGenerator(self._model)
        self._predictor = SamPredictor(self._model)
        self._device = device

    def __call__(self, 
                 image: ImageLike | list[ImageLike],
                 prompt_boxes: list[BoundingBox] | list[list[BoundingBox]] | None = None,
                 return_boxes: bool = True,
                 return_masks: bool = True,
                 return_crops: bool = True,
                 return_crops_no_background: bool = True) -> Segment | list[Segment]:
        """
        Extract segments from the image.

        Args:
            image: The image (or list of images) to extract segments from.

        Returns:
            A list of segments (or list of lists of segments if a list of images is provided).
        """
        if isinstance(image, list):
            if prompt_boxes is not None:
                assert len(prompt_boxes) == len(image), \
                    "If prompt_boxes are provided, they must be the same length as the image"
            else:
                prompt_boxes = [None] * len(image)
            return [self._process_image(img, 
                                        prompt_boxes=prompt_boxes[i],
                                        return_boxes=return_boxes,
                                        return_masks=return_masks,
                                        return_crops=return_crops,
                                        return_crops_no_background=return_crops_no_background) 
                    for i, img in enumerate(image)]
        else:
            return self._process_image(image, 
                                       prompt_boxes=prompt_boxes,
                                       return_boxes=return_boxes,
                                       return_masks=return_masks,
                                       return_crops=return_crops,
                                       return_crops_no_background=return_crops_no_background)

    def _process_image(self, 
                       image: ImageLike,
                       prompt_boxes: list[BoundingBox] | None = None,
                       return_boxes: bool = True,
                       return_masks: bool = True,
                       return_crops: bool = True,
                       return_crops_no_background: bool = True) -> list[Segment]:
        image = self._load_image(image)
        result = []

        if prompt_boxes is None:
            sam_output = self._mask_generator.generate(image)
            for segment in sam_output:
                filtered_image = image * np.tile(segment["segmentation"][:, :, np.newaxis], (1, 1, 3))
                x, y, w, h = segment["bbox"]
                kwargs = {}
                if return_boxes:
                    kwargs["bbox"] = BoundingBox(xywh=(x, y, w, h))
                if return_masks:
                    kwargs["mask"] = segment["segmentation"]
                if return_crops:
                    kwargs["crop"] = image[y:y+h, x:x+w]
                if return_crops_no_background:
                    kwargs["crop_no_background"] = filtered_image[y:y+h, x:x+w]
                result.append(Segment(**kwargs))

        else:
            self._predictor.set_image(image)
            input_boxes = torch.tensor([
                e.xyxy for e in prompt_boxes], device=self._device)
            transformed_boxes = self._predictor.transform.apply_boxes_torch(input_boxes, 
                                                                            image.shape[:2])
            masks, _, _ = self._predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            masks = masks.cpu().numpy()
            for i in range(masks.shape[0]):
                mask = masks[i, 0]
                filtered_image = image * np.tile(mask[:, :, np.newaxis], (1, 1, 3))
                x, y, w, h = prompt_boxes[i].xywh
                kwargs = {}
                if return_boxes:
                    kwargs["bbox"] = prompt_boxes[i]
                if return_masks:
                    kwargs["mask"] = mask
                if return_crops:
                    kwargs["crop"] = image[y:y+h, x:x+w]
                if return_crops_no_background:
                    kwargs["crop_no_background"] = filtered_image[y:y+h, x:x+w]
                result.append(Segment(**kwargs))
        
        return result

    def _load_image(self, image: ImageLike) -> np.ndarray:
        if isinstance(image, str) or isinstance(image, Path):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError(f"Invalid image-like type: {type(image)}")
        return image


class SAMTraversePredictor:
    """
    Wrapper over the Segment Anything Model's automatic mask generator.
    """

    def __init__(self,
                 model_type: SAMModelType,
                 checkpoint_path: PathLike,
                 criterion: MaskCriterion | str | list[MaskCriterion] | list[str],
                 device: str = "cuda"):
        """
        Initialize the SAM model for traverse mask generation.

        Args:
            model_type: The type of SAM model to use.
            checkpoint_path: The path to the SAM checkpoint.
            device: The device to run the model on.
        """
        if isinstance(model_type, str):
            model_type = SAMModelType(model_type)
        if isinstance(criterion, str):
            criterion = [MaskCriterion(criterion)]
        elif isinstance(criterion, list):
            if isinstance(criterion[0], str):
                criterion = [MaskCriterion(c) for c in criterion]
            else:
                criterion = criterion
        elif isinstance(criterion, MaskCriterion):
            criterion = [criterion]
        else:
            raise ValueError(f"Invalid criterion: {criterion}")

        self._criterion = criterion
        self._model = sam_model_registry[model_type.value](checkpoint=str(checkpoint_path)).to(device)
        self._predictor = SamPredictor(self._model)
        self._device = device

    def __call__(self, 
                 image: ImageLike | list[ImageLike],
                 waypoint_pixels: np.ndarray | list[np.ndarray]) -> Segment | list[Segment]:
        """
        Extract segments from the image.

        Args:
            image: The image (or list of images) to extract segments from.

        Returns:
            A list of segments (or list of lists of segments if a list of images is provided).
        """
        if isinstance(image, list):
            assert isinstance(waypoint_pixels, list) and len(waypoint_pixels) == len(image), \
                "waypoint_pixels must be a list of numpy arrays with the same length as the image"
            return [self._process_image(img, 
                                        waypoint_pixels[i]) 
                    for i, img in enumerate(image)]
        else:
            return self._process_image(image, 
                                       waypoint_pixels=waypoint_pixels)

    def _process_image(self, 
                       image: ImageLike,
                       waypoint_pixels: np.ndarray) -> list[Segment]:
        image = self._load_image(image)
        result = []

        self._predictor.set_image(image)
        masks, scores, logits = self._predictor.predict(
            point_coords=waypoint_pixels,
            point_labels=np.ones(waypoint_pixels.shape[0]),
            multimask_output=True,
        )

        result = {}
        for criterion in self._criterion:
            if criterion == MaskCriterion.SCORE:
                result[str(criterion.value)] = masks[np.argmax(scores)].copy()
            elif criterion == MaskCriterion.AREA:
                result[str(criterion.value)] = masks[np.argmax([e.sum() for e in masks])].copy()
            elif criterion == MaskCriterion.ALL:
                result[str(criterion.value)] = masks.copy()
            else:
                raise ValueError(f"Invalid criterion: {self._criterion}")
        return result

    def _load_image(self, image: ImageLike) -> np.ndarray:
        if isinstance(image, str) or isinstance(image, Path):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError(f"Invalid image-like type: {type(image)}")
        return image
