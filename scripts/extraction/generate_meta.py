import autoroot
import json
import fire

from pathlib import Path
from typing import Dict
from egowalk_pipelines.utils.camera_utils import DEFAULT_CAMERA_PARAMS
from egowalk_pipelines.extraction.locators import locate_svo_files


def get_all_heights(raw_root: Path) -> Dict[str, float]:
    svo_files = locate_svo_files(raw_root)
    heights = {}
    for svo_file in svo_files:
        svo_dir = svo_file.parent
        with open(svo_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        height = metadata["height"]
        heights[svo_dir.stem] = height / 100.
    return heights


def fill_info(meta_dir: Path):
    with open(meta_dir / "info.json", "w") as f:
        json.dump({"fps": 5}, f, indent=2)


def fill_camera_params(meta_dir: Path):
    camera_params = {
        "fx": DEFAULT_CAMERA_PARAMS._fx,
        "fy": DEFAULT_CAMERA_PARAMS._fy,
        "cx": DEFAULT_CAMERA_PARAMS._cx,
        "cy": DEFAULT_CAMERA_PARAMS._cy,
        "k1": DEFAULT_CAMERA_PARAMS._k1,
        "k2": DEFAULT_CAMERA_PARAMS._k2,
        "p1": DEFAULT_CAMERA_PARAMS._p1,
        "p2": DEFAULT_CAMERA_PARAMS._p2,
        "k3": DEFAULT_CAMERA_PARAMS._k3,
        "k4": DEFAULT_CAMERA_PARAMS._k4,
        "k5": DEFAULT_CAMERA_PARAMS._k5,
        "k6": DEFAULT_CAMERA_PARAMS._k6,
    }
    with open(meta_dir / "camera_rgb.json", "w") as f:
        json.dump(camera_params, f, indent=2)


def fill_trajectories(output_root: Path,
                      meta_dir: Path):
    trajectories = sorted([e.stem for e in (output_root / "data").glob("*.parquet")])
    with open(meta_dir / "trajectories.json", "w") as f:
        json.dump(trajectories, f, indent=2)


def fill_heights(raw_root: Path,
                 output_root: Path,
                 meta_dir: Path):
    trajectories = sorted([e.stem for e in (output_root / "data").glob("*.parquet")])
    all_heights = get_all_heights(raw_root)
    traj_heights = {e: all_heights[e] for e in trajectories}
    with open(meta_dir / "heights.json", "w") as f:
        json.dump(traj_heights, f, indent=2)


def main(raw_root: str,
         output_root: str):
    raw_root = Path(raw_root)
    output_root = Path(output_root)

    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    fill_info(meta_dir)
    fill_camera_params(meta_dir)
    fill_trajectories(output_root, meta_dir)
    fill_heights(raw_root, output_root, meta_dir)



if __name__ == "__main__":
    fire.Fire(main)
