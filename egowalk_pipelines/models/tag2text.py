import enum
import numpy as np
import torch
import torchvision.transforms as TorchTransforms

from pathlib import Path
from PIL import Image
from ram.models import tag2text
from egowalk_pipelines.misc.types import PathLike, ImageLike




class Tag2Text:
    """
    Recognize Anything Model (RAM) wrapper.
    """
    _IMAGE_SIZE = 384
    _VIT = "swin_l"

    def __init__(self,
                 model_checkpoint: PathLike,
                 device: str = "cuda"):
        """
        Initialize the Tag2Text model.

        Args:
            model_checkpoint: The path to the model checkpoint.
            device: The device to run the model on.
        """
        delete_tag_index = []
        for i in range(3012, 3429):
            delete_tag_index.append(i)
        self._model = tag2text(pretrained=str(model_checkpoint),
                               image_size=384,
                               vit='swin_b',
                               delete_tag_index=delete_tag_index)
        self._model.eval()
        self._model.to(device)
        self._device = device

        self._transform = TorchTransforms.Compose([
            TorchTransforms.Resize((384, 384)),
            TorchTransforms.ToTensor(),
            TorchTransforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image: ImageLike | list[ImageLike]) -> list[str] | list[list[str]]:
        """
        Generate tags for an image or a list of images.

        Args:
            image: The image or list of images to generate tags for.

        Returns:
            A list of tags for the image or a list of lists of tags for a list of images.
        """
        if not isinstance(image, list):
            image = self._load_image(image)
            with torch.no_grad():
                _, tags = self._model.generate(image.unsqueeze(0).to(self._device),
                                              tag_input=None,
                                              max_length=50,
                                              return_tag_predict=True)
                tags = tags[0]
                tags = self._split_tags(tags)
        else:
            image = torch.stack([self._load_image(img) for img in image],
                                dim=0).to(self._device)
            with torch.no_grad():
                _, tags = self._model.generate(image,
                                              tag_input=None,
                                              max_length=50,
                                              return_tag_predict=True)
                tags = [self._split_tags(tag) for tag in tags]
        return tags

    def _load_image(self, image: ImageLike) -> torch.Tensor:
        if isinstance(image, str) or isinstance(image, Path):
            image = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pass
        else:
            raise ValueError(f"Invalid image type: {type(image)}")
        image = image.resize((384, 384))
        image = self._transform(image)
        return image

    def _split_tags(self, tags: str) -> list[str]:
        tags = tags.strip(' ').replace('  ', ' ').replace(' |', ',').split(',')
        tags = [tag.strip() for tag in tags]
        return tags
