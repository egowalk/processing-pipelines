import numpy as np
import torch
import torchvision
import groundingdino.datasets.transforms as T

from pathlib import Path
from PIL import Image
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from egowalk_pipelines.misc.types import ImageLike, PathLike
from egowalk_pipelines.misc.segments import BoundingBox


class GroundingDINOModel:
    """
    GroundingDINO model wrap for prompt-based detection.
    """
    
    def __init__(self,
                 config: PathLike,
                 model_checkpoint: PathLike,
                 box_threshold: float = 0.25,
                 text_threshold: float = 0.2,
                 iou_threshold = 0.5,
                 device: str = "cuda",
                 use_nms: bool = False) -> None:
        """
        Initialize GroundingDINO model.

        Args:
            config: Path to the config file.
            model_checkpoint: Path to the model checkpoint.
            box_threshold: Threshold for the bounding box.
            text_threshold: Threshold for the text.
        """
        args = SLConfig.fromfile(config)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint, map_location="cpu")
        load_res = model.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False)
        _ = model.eval()
        model.to(device)
        self._model = model
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._iou_threshold = iou_threshold
        self._device = device
        self._use_nms = use_nms
        
        self._transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __call__(self, 
                 image: ImageLike | list[ImageLike],
                 tags: list[str] | list[list[str]]) -> tuple[BoundingBox, list[str]] |\
                      list[tuple[BoundingBox, list[str]]]:
        """
        Generate bounding boxes and labels for the given image(s) and tags.

        Args:
            image: Image or list of images.
            tags: List of tags or list of lists of tags.

        Returns:
            BoundingBox or list of bounding boxes.
            List of labels or list of lists of labels.
        """
        if isinstance(image, list):
            if not isinstance(tags[0], list):
                tags = [tags] * len(image)
            else:
                assert len(image) == len(tags)
            result = [self._process_image(img, tag) for img, tag in zip(image, tags)]
            return [result[i][0] for i in range(len(result))], [result[i][1] for i in range(len(result))]
        else:
            if not isinstance(tags[0], list):
                return self._process_image(image, tags)
            else:
                image = [image] * len(tags)
                result = [self._process_image(img, tag) for img, tag in zip(image, tags)]
                return [result[i][0] for i in range(len(result))], [result[i][1] for i in range(len(result))]

    def _process_image(self, 
                       image: ImageLike,
                       tags: list[str]) -> tuple[BoundingBox, list[str]]:
        image, (H, W)= self._load_image(image)
        
        caption = ", ".join(tags)
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."

        with torch.no_grad():
            outputs = self._model(image.unsqueeze(0).to(self._device), 
                                  captions=[caption])
        
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self._box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = self._model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > self._text_threshold, tokenized, tokenlizer)
            # pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            pred_phrases.append(pred_phrase)
            scores.append(logit.max().item())
        scores = torch.Tensor(scores)
        
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        # use NMS to handle overlapped 
        if self._use_nms:
            nms_idx = torchvision.ops.nms(boxes_filt, scores, self._iou_threshold).numpy().tolist()
            boxes_filt = boxes_filt[nms_idx]
            pred_phrases = [pred_phrases[idx] for idx in nms_idx]

        boxes_filt = self._convert_boxes(boxes_filt, (W, H))

        # TODO: Return scores?
        return boxes_filt, pred_phrases

    def _load_image(self, image: ImageLike) -> torch.Tensor:
        if isinstance(image, str) or isinstance(image, Path):
            image = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pass
        else:
            raise ValueError(f"Invalid image type: {type(image)}")
        image_size = image.size
        image, _ = self._transform(image, None)
        return image, (image_size[1], image_size[0])

    def _convert_boxes(self, 
                       boxes: torch.Tensor, 
                       image_wh: tuple[int, int]) -> list[BoundingBox]:
        W, H = image_wh
        boxes = boxes.numpy()
        result = []
        for box in boxes:
            box = np.clip(box, np.array([0, 0, 0, 0]), np.array([W, H, W, H]))
            box = box.round().astype(int)
            result.append(BoundingBox.from_xyxy(tuple(box)))
        return result
