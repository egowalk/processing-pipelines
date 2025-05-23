import yaml
import json
import fire 
import numpy as np
import matplotlib.pyplot as plt
import cv2

from pathlib import Path
from typing import Any
from functools import partial
from tqdm import tqdm
from egowalk_pipelines.models.sam import SAMTraversePredictor
from egowalk_pipelines.utils.camera_utils import DEFAULT_CAMERA_PARAMS, project_points
from egowalk_pipelines.utils.parallel_utils import do_parallel
from egowalk_pipelines.utils.str_utils import zfill_zeros_pad
from egowalk_dataset.datasets.gnm.gnm_indexing import index_gnm
from egowalk_dataset.datasets.gnm.cutters import (SpikesCutter,
                                                  StuckCutter,
                                                  BackwardCutter)
from egowalk_dataset.datasets.base.base_dataset import EgoWalkBaseDataset
from egowalk_dataset.datasets.gnm.gnm_dataset import GNMDataset, DefaultGNMDataset, GNMRGBFeature, GNMDepthFeature, GNMWaypointFeature, GNMFeature, GNMTuple


IMAGE_WIDTH = 960
IMAGE_HEIGHT = 600


DIR_IMAGE = "image"
DIR_MASK_SCORE = "mask_score"
DIR_MASK_AREA = "mask_area"

N_ZEROS_PAD = 7


class GNMHeightFeature(GNMFeature):

    def __init__(self,
                 name: str,
                 heights: dict[str, float]):
        super(GNMHeightFeature, self).__init__(name)
        self._heights = heights

    def __call__(self,
                 root: Path,
                 gnm_tuple: GNMTuple) -> float:
        height = self._heights[gnm_tuple.trajectory_name]
        return height
    

class GNMFrameIdxFeature(GNMFeature):

    def __init__(self,
                 name: str):
        super(GNMFrameIdxFeature, self).__init__(name)

    def __call__(self,
                 root: Path,
                 gnm_tuple: GNMTuple) -> float:
        return gnm_tuple.obs_idxs[-1]
    

class GNMTrajNameFeature(GNMFeature):

    def __init__(self,
                 name: str):
        super(GNMTrajNameFeature, self).__init__(name)

    def __call__(self,
                 root: Path,
                 gnm_tuple: GNMTuple) -> float:
        return gnm_tuple.trajectory_name


def _split_list(source_list: list[Any],
                n_chunks: int) -> list[list[Any]]:
    if n_chunks == 0:
        n_chunks = 1
    chunk_size = len(source_list) // n_chunks
    chunks = [source_list[i:i + chunk_size] for i in range(0, len(source_list), chunk_size)]
    return chunks


def _create_index(dataset_root: Path,
                  config: dict[str, Any]) -> dict[str, Any]:
    gnm_index = index_gnm(cutters=[StuckCutter(eps=1e-2),
                               BackwardCutter(backwards_eps=1e-2,
                                              stuck_eps=1e-2,
                                              ignore_stuck=True),
                                SpikesCutter(spike_threshold=2.)],
                      window_step=config["window_step"],
                      context_length=1,
                      goal_offset=7,
                      goal_offset_mode="fixed",
                      action_length=config["action_length"],
                      context_step=1,
                      action_step=config["action_step"],
                      root=dataset_root,
                      n_workers=12,
                      use_tqdm=False)
    return gnm_index


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.uint8(mask) * 255
    return mask


def _process_index(index: dict[str, Any],
                   dataset_root: Path,
                   heights: dict[str, float],
                   min_distance: float,
                   weights: Path,
                   output_dir: Path) -> dict[str, Any]:
    ds = GNMDataset(index=index,
                 root=dataset_root,
                 features=[GNMRGBFeature(name="obs",
                                        field="obs",
                                        indices=-1),
                            GNMDepthFeature(name="obs_depth",
                                            field="obs",
                                            indices=-1),
                            GNMWaypointFeature(name="action",
                                               angle_format="none",
                                               return_tensors="np"),
                            GNMHeightFeature(name="height",
                                             heights=heights),
                            GNMFrameIdxFeature(name="frame_idx"),
                            GNMTrajNameFeature(name="traj_name")])
    
    traverse_predictor = SAMTraversePredictor(model_type="vit_h",
                                              checkpoint_path=str(weights),
                                              criterion=["score", "area"])
    
    for i in tqdm(range(len(ds))):
        item = ds[i]
        img = item["obs"]
        traj = item["action"]
        height = item["height"]
        frame_idx = item["frame_idx"]
        traj_name = item["traj_name"]
        depth_img = item["obs_depth"]

        pixels = project_points(traj,
                                height,
                                DEFAULT_CAMERA_PARAMS,
                                (IMAGE_WIDTH, IMAGE_HEIGHT))
        
        if pixels is None:
            continue

        masks = traverse_predictor(img, pixels)
        mask_score = masks["score"]
        mask_area = masks["area"]

        depth_img[np.isnan(depth_img)] = 20.
        depth_masked = depth_img[mask_score]
        min_mask_distance = depth_masked.min()
        if min_mask_distance < min_distance:
            continue
        depth_masked = depth_img[mask_area]
        min_mask_distance = depth_masked.min()
        if min_mask_distance < min_distance:
            continue
        
        file_name = f"{traj_name}__{zfill_zeros_pad(frame_idx, N_ZEROS_PAD)}"
        output_img = output_dir / DIR_IMAGE / f"{file_name}.jpg"
        output_mask_score = output_dir / DIR_MASK_SCORE / f"{file_name}.png"
        output_mask_area = output_dir / DIR_MASK_AREA / f"{file_name}.png"
        
        cv2.imwrite(str(output_img), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(str(output_mask_score), _transform_mask(mask_score))
        cv2.imwrite(str(output_mask_area), _transform_mask(mask_area))


def main(dataset_root: str,
         config: str,
         output_dir: str,
         weights: str,
         n_workers: int = 0):
    dataset_root = Path(dataset_root)
    config = Path(config)
    output_dir = Path(output_dir)
    weights = Path(weights)

    if output_dir.is_dir():
        raise ValueError(f"Output directory {output_dir} already exists")

    with open(config, 'r') as f:
        config = yaml.safe_load(f)

    gnm_index = _create_index(dataset_root, config)
    indices = []
    for k, v in gnm_index.items():
        splits = _split_list(v, n_workers)
        if len(indices) == 0:
            for split in splits:
                indices.append({k: split})
        else:
            for split, index in zip(splits, indices):
                index[k] = split

    with open(dataset_root / "meta" / "heights.json", "r") as f:
        heights = json.load(f)
    
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / DIR_IMAGE).mkdir(parents=True, exist_ok=False)
    (output_dir / DIR_MASK_SCORE).mkdir(parents=True, exist_ok=False)
    (output_dir / DIR_MASK_AREA).mkdir(parents=True, exist_ok=False)
    
    task_fn = partial(_process_index,
                       dataset_root=dataset_root,
                       heights=heights,
                       min_distance=config["min_distance"],
                       weights=weights,
                       output_dir=output_dir)
    do_parallel(task_fn, indices, n_workers, use_tqdm=False)


if __name__ == "__main__":
    fire.Fire(main)
