import numpy as np
import cv2

from egowalk_pipelines.misc.segments import BoundingBox


class FaceBlurrer:

    def __init__(self,
                 kernel: int | str | None = "auto",
                 sigma: float | None = None):
        if kernel is None:
            self._kernel = None
        elif isinstance(kernel, str):
            if kernel == "auto":
                self._kernel = None
            else:
                raise ValueError(f"Invalid kernel: {kernel}")
        elif isinstance(kernel, int):
            if kernel % 2 == 0:
                kernel = kernel + 1
            self._kernel = kernel
        else:
            raise ValueError(f"Invalid kernel: {kernel}")
        
        if sigma is None:
            self._sigma = 0.
        elif isinstance(sigma, float):
            assert sigma >= 0, f"Sigma must be non-negative, got {sigma}"
            self._sigma = sigma
        else:
            raise ValueError(f"Invalid sigma: {sigma}")

    def __call__(self, 
                 image: np.ndarray,
                 boxes: list[BoundingBox] | None) -> np.ndarray:
        image = image.copy()
        if boxes is None or len(boxes) == 0: 
            return image
        
        if self._sigma is not None:
            sigma = self._sigma
        else:
            sigma = 0.

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy

            if self._kernel is None:
                _, _, w, h = box.xywh
                kernel_size = max(w, h)
                if kernel_size % 2 == 0:
                    kernel_size = kernel_size + 1
            else:
                kernel_size = self._kernel
            
            image[y1:y2, x1:x2] = cv2.GaussianBlur(image[y1:y2, x1:x2], (kernel_size, kernel_size), sigmaX=sigma)
            
        return image
