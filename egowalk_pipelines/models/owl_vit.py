import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor
from egowalk_pipelines.misc.types import ImageLike
from egowalk_pipelines.misc.segments import BoundingBox


class OWLViT:

    def __init__(self,
                 prompt: str,
                 box_threshold: float,
                 text_threshold: float,
                 device: str = "cuda"):
        self._device = device
        self._prompt = prompt
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self._model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(self._device)
        self._model.eval()

    def __call__(self, images: ImageLike | list[ImageLike]) -> list[BoundingBox] | list[list[BoundingBox]]:
        if not isinstance(images, list):
            images = [images]
            single_image = True
        else:
            single_image = False

        images = [self._load_image(e) for e in images]
        model_input = self._processor(text=[self._prompt] * len(images), 
                                      images=images, 
                                      return_tensors="pt").to(self._device)
        with torch.inference_mode():
            outputs = self._model(**model_input)

        target_sizes = torch.tensor([[e.size[1], e.size[0]] for e in images], dtype=torch.float32).to(self._device)  # [height, width]
        results = self._processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes,
            threshold=self._box_threshold  # Assuming you want to use the same threshold
        )

        all_boxes = []
        for result in results:
            boxes = result["boxes"].cpu().numpy() 
            boxes = [box for box in boxes if (box >= 0.).all()]
            boxes = [BoundingBox.from_xyxy((int(box[0]), int(box[1]), int(box[2]), int(box[3]))) for box in boxes]
            all_boxes.append(boxes)
        
        return all_boxes if not single_image else all_boxes[0]
        

    def _load_image(self, image: ImageLike) -> Image.Image:
        if isinstance(image, str) or isinstance(image, Path):
            return Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image)
        elif isinstance(image, Image.Image):
            return image
        else:
            raise ValueError(f"Invalid image type: {type(image)}")


class OWLVitFaceDetector(OWLViT):

    def __init__(self,
                 prompt: str = "human face",
                 box_threshold: float = 0.1,
                 text_threshold: float = 0.2,
                 device: str = "cuda"):
        super(OWLVitFaceDetector, self).__init__(prompt, box_threshold, text_threshold, device)

    def __call__(self, images: ImageLike | list[ImageLike]) -> list[BoundingBox] | list[list[BoundingBox]]:
        return super(OWLVitFaceDetector, self).__call__(images)
