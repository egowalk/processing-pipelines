import autoroot
import yaml
import fire
import luigi
import numpy as np
from typing import List, Tuple, Any
from pathlib import Path
from egowalk_dataset.datasets.trajectory.trajectory import EgoWalkTrajectory
from egowalk_pipelines.misc.segments import BoundingBox
from egowalk_pipelines.utils.math_utils import to_relative_frame
from egowalk_pipelines.models.ram import RAM
from egowalk_pipelines.models.grounding_dino import GroundingDINOModel
from egowalk_pipelines.models.segments_extraction import RAMGroundingDINOSegmentsExtractor
from egowalk_pipelines.annotation.tools import SceneObjectFinder, SceneObjectSelector
from egowalk_pipelines.annotation.filters import AreaObjectsFilter, CloseObjectsFilter, FarTrajObjectsFilter, ClosestTrajObjectSelector
from egowalk_pipelines.utils.extracted_trajectory_utils import split_traj
from egowalk_pipelines.utils.sync_utils import TimestampIndexer
from egowalk_pipelines.utils.io_utils import SequentialCSVWriter


_HOST_DOCKER_HOST = "docker_host"

_ANNOTATION_FILE_NAME = "annotation_boxes.csv"


class ProcessSingleTrajectory(luigi.Task):
    data_root = luigi.Parameter(description="Path to the data root directory")
    traj_name = luigi.Parameter(description="Name of the trajectory to process")
    output_root = luigi.Parameter(description="Path to the output root directory")
    config = luigi.Parameter(description="Path to the config file")

    def output(self):
        annotation_file = Path(self.output_root) / self.traj_name / _ANNOTATION_FILE_NAME
        return luigi.LocalTarget(str(annotation_file))

    def run(self):
        print(f"Processing {self.traj_name}...")
        config = self._load_config()
        objects = self._extract_objects(config)
        self._dump_objects(objects)

    def _extract_objects(self, config: dict[str, Any]) -> List[Tuple[str, BoundingBox]]:
        segments_extractor = RAMGroundingDINOSegmentsExtractor(RAM(model_version=config["segments_extractor"]["ram"]["model_version"],
                                                                   model_checkpoint=config["segments_extractor"]["ram"]["model_checkpoint"],
                                                                   device=config["segments_extractor"]["device"]),
                                                               GroundingDINOModel(config=config["segments_extractor"]["grounding_dino"]["config"],
                                                                                  model_checkpoint=config["segments_extractor"]["grounding_dino"]["model_checkpoint"],
                                                                                  device=config["segments_extractor"]["device"]),
                                                               None,
                                                               config["segments_extractor"]["tags_blacklist"],
                                                               config["segments_extractor"]["stochastic_tags_blacklist"])
        scene_object_finder = SceneObjectFinder(segments_extractor)
        scene_object_selector = SceneObjectSelector(filters=[
                               AreaObjectsFilter(threshold_percent=config["scene_object_selector"]["area_objects_filter"]["threshold_percent"]),
                               CloseObjectsFilter(min_distance=config["scene_object_selector"]["close_objects_filter"]["min_distance"]),
                               FarTrajObjectsFilter(max_distance=config["scene_object_selector"]["far_traj_objects_filter"]["max_distance"])
                           ],
                           selector=ClosestTrajObjectSelector())
        
        splits = split_traj(self.traj_name,
                            self.data_root,
                            distance_threshold=config["split_traj"]["distance_threshold"])
        start_ts = splits[0][0]
        end_ts = splits[-1][-1]
        result = []
        for split in splits:
            result.extend(self._extract_from_split(split, 
                                                   config, 
                                                   scene_object_finder, 
                                                   scene_object_selector,
                                                   (start_ts, end_ts)))
        self._progress_monitor(end_ts, (start_ts, end_ts))
        return result
    
    def _extract_from_split(self, 
                            timestamps: List[Path], 
                            config: dict[str, Any],
                            scene_object_finder: SceneObjectFinder,
                            scene_object_selector: SceneObjectSelector,
                            global_start_end_timestamps: Tuple[int, int]) -> List[Tuple[str, BoundingBox, str]]:
        traj = EgoWalkTrajectory.from_dataset(self.traj_name,
                                              self.data_root,
                                              timestamps=timestamps)

        rgb_timestamps = traj.rgb.timestamps
        odom_timestamps = traj.odometry.valid_timestamps
        odom_indexer = TimestampIndexer(odom_timestamps)
        traj_2d = traj.odometry.get_bev(filter_valid=True)

        result = []

        for i in range(0, len(rgb_timestamps), config["global_traj_interval"]):
            rgb_ts = rgb_timestamps[i]
            odom_ts, odom_idx = odom_indexer.query(rgb_ts)

            footsteps = traj_2d[odom_idx:(odom_idx + config["local_traj_horizon"]):config["local_traj_interval"], :]
            footsteps = footsteps[~np.isnan(footsteps).any(axis=1)]
            if footsteps.shape[0] < config["min_local_traj_length"]:
                continue
            
            footsteps = to_relative_frame(footsteps)
            img = traj.rgb.at(rgb_ts)
            depth_img = traj.depth.at(rgb_ts)

            objects = scene_object_finder(img, depth_img)
            
            goal_object = scene_object_selector(objects=objects,
                                                context={"trajectory_bev": footsteps,
                                                         "image_wh": (img.shape[1], img.shape[0])})
        
            self._progress_monitor(rgb_ts, global_start_end_timestamps)
            if goal_object is None:
                continue
            result.append((rgb_ts, goal_object.segment.bbox, goal_object.segment.tags[0]))

        return result
    
    def _dump_objects(self, objects: List[Tuple[str, BoundingBox, str]]):
        csv_writer = SequentialCSVWriter(columns=["timestamp", "box_x", "box_y", "box_w", "box_h", "tag"])
        for rgb_ts, bbox, tag in objects:
            x, y, w, h = bbox.xywh
            csv_writer.add_line((rgb_ts, x, y, w, h, tag))
        output_dir = Path(self.output().path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_writer.dump(self.output().path)

    def _load_config(self) -> dict[str, Any]:
        config_path = Path(self.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file {config_path} does not exist")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _progress_monitor(self,
                          current_timestamp: int,
                          global_start_end_timestamps: Tuple[int, int]):
        if current_timestamp == global_start_end_timestamps[1]:
            self.set_progress_percentage(100)
            return
        ts_relative = current_timestamp - global_start_end_timestamps[0]
        duration = global_start_end_timestamps[1] - global_start_end_timestamps[0]
        self.set_progress_percentage(round((ts_relative / duration) * 100, 2))


class ProcessAllTrajectories(luigi.WrapperTask):
    data_root = luigi.Parameter(description="Path to the data root directory")
    output_root = luigi.Parameter(description="Path to the output root directory")
    config = luigi.Parameter(description="Path to the config file")

    def requires(self):
        trajectories = self._read_list()
        for trajectory in trajectories:
            yield ProcessSingleTrajectory(
                data_root=self.data_root,
                traj_name=trajectory,
                output_root=self.output_root,
                config=self.config
            )

    def _read_list(self) -> List[Path]:
        trajectories_root = Path(self.data_root)
        trajectories = sorted([e.stem for e in (trajectories_root / "data").glob("*.parquet")])
        return trajectories


def main(data_root: str,
         output_root: str,
         config: str,
         n_workers: int = 0,
         local: bool = False,
         scheduler_host: str = "localhost",
         scheduler_port: int = 8082):
    data_root = Path(data_root)
    output_root = Path(output_root)
    config = Path(config)

    kwargs = {}
    if local:
        kwargs["local_scheduler"] = True
    else:
        if scheduler_host == _HOST_DOCKER_HOST:
            scheduler_host = "172.17.0.1"
        kwargs["scheduler_host"] = scheduler_host
        kwargs["scheduler_port"] = scheduler_port
    if n_workers > 0:
        kwargs["workers"] = n_workers

    luigi.build([
        ProcessAllTrajectories(
            data_root=str(data_root),
            output_root=str(output_root),
            config=str(config)
        )],
        **kwargs
    )


if __name__ == "__main__":
    fire.Fire(main)
