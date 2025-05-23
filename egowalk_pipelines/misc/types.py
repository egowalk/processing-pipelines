import numpy as np

from typing import Union, Tuple
from pathlib import Path
from PIL import Image


ImageLike = Union[str, Path, np.ndarray, Image.Image]

PathLike = Union[str, Path]

RotTrans = Tuple[float, float, float, float, float, float, float]
