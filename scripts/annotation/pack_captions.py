import json
import fire
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pathlib import Path
from functools import partial
from egowalk_dataset.datasets.trajectory.trajectory import EgoWalkTrajectory
from egowalk_pipelines.utils.parallel_utils import do_parallel


def process_trajectory_annotations(traj_dir: Path,
                                   target_trajs: list[str],
                                   data_root: Path,
                                   subset: str,
                                   output_root: Path):
    traj_name = traj_dir.name
    if traj_name not in target_trajs:
        return
    
    output_parquet = output_root / f"{traj_name}.parquet"
    if output_parquet.exists():
        print(f"Trajectory {traj_name} already exists")
        return

    csv_file_name = "filtered_captions_normal_v2.csv"
    output_parquet = output_root / "annotations" / "normal" / f"{traj_name}__annotations_normal.parquet"

    source_df = pd.read_csv(traj_dir / csv_file_name)

    traj = EgoWalkTrajectory.from_dataset(traj_name, data_root)
    source_timestamps = traj.rgb.timestamps
    
    target_timestamps = []
    target_indices = []
    target_captions = []
    target_box_x = []
    target_box_y = []
    target_box_w = []
    target_box_h = []

    for _, row in source_df.iterrows():
        timestamp = row["timestamp"]
        idx = source_timestamps.index(timestamp)
        box_x = row["box_x"]
        box_y = row["box_y"]
        box_w = row["box_w"]
        box_h = row["box_h"]
        caption = row["caption"]

        target_timestamps.append(timestamp)
        target_indices.append(idx)
        target_captions.append(caption)
        target_box_x.append(box_x)
        target_box_y.append(box_y)
        target_box_w.append(box_w)
        target_box_h.append(box_h)

    dataset_dict = {
        "timestamp": target_timestamps,
        "trajectory": [traj_name] * len(target_timestamps),
        "frame": target_indices,
        "caption": target_captions,
        "box_x": target_box_x,
        "box_y": target_box_y,
        "box_w": target_box_w,
        "box_h": target_box_h,
    }
    
    df = pd.DataFrame(dataset_dict)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, str(output_parquet))


def process_trajectory(traj_dir: Path,
                       target_trajs: list[str],
                       output_root: Path):
    process_trajectory_annotations(traj_dir, target_trajs, "normal", output_root)
    process_trajectory_annotations(traj_dir, target_trajs, "brief", output_root)


def main(annotation_dir: str,
         data_root: str,
         n_workers: int = 0):
    annotation_dir = Path(annotation_dir)
    data_root = Path(data_root)

    trajectory_dirs = sorted(annotation_dir.glob("*/"))

    (data_root / "annotations/normal").mkdir(parents=True, exist_ok=True)

    with open(data_root / "meta/trajectories.json", "r") as f:
        filtered_trajs = json.load(f)

    task_fn = partial(process_trajectory,
                      target_trajs=filtered_trajs,
                      data_root=data_root,
                      output_root=data_root)
    do_parallel(task_fn,
                trajectory_dirs,
                n_workers=n_workers,
                use_tqdm=True)
    

if __name__ == "__main__":
    fire.Fire(main)
