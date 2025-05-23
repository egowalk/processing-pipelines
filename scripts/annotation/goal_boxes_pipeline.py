import yaml
import fire
import luigi
import numpy as np
from typing import List, Union, Tuple, Any
from pathlib import Path
from egowalk_tools.trajectory import DefaultTrajectory
from canguro_processing_tools.misc.segments import BoundingBox
from canguro_processing_tools.utils.camera_utils import (DEFAULT_CAMERA_PARAMS,
                                                         project_points)
from canguro_processing_tools.utils.math_utils import to_relative_frame
from canguro_processing_tools.models.sam import SAM
from canguro_processing_tools.misc.segments import Segment, BoundingBox
from canguro_processing_tools.models.ram import RAM
from canguro_processing_tools.models.tag2text import Tag2Text
from canguro_processing_tools.models.grounding_dino import GroundingDINOModel
from canguro_processing_tools.models.segments_extraction import RAMGroundingDINOSegmentsExtractor, DEFAULT_TAGS_BLACKLIST
from canguro_processing_tools.utils.str_utils import seconds_to_human_readable_time
from canguro_processing_tools.annotation.tools import SceneObjectFinder, SceneObjectSelector
from canguro_processing_tools.annotation.filters import AreaObjectsFilter, CloseObjectsFilter, FarTrajObjectsFilter, ClosestTrajObjectSelector
from canguro_processing_tools.utils.extracted_trajectory_utils import split_traj
from canguro_processing_tools.utils.sync_utils import TimestampIndexer
from canguro_processing_tools.utils.io_utils import SequentialCSVWriter


_HOST_DOCKER_HOST = "docker_host"

_ANNOTATION_FILE_NAME = "annotation_boxes.csv"


class ProcessSingleTrajectory(luigi.Task):
    trajectory_dir = luigi.Parameter(description="Path to the trajectory directory to process")
    config = luigi.Parameter(description="Path to the config file")

    def output(self):
        trajectory_dir = Path(self.trajectory_dir)
        annotation_file = trajectory_dir / _ANNOTATION_FILE_NAME
        return luigi.LocalTarget(str(annotation_file))

    def run(self):
        print(f"Processing {self.trajectory_dir}...")
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
        
        splits = split_traj(self.trajectory_dir, distance_threshold=config["split_traj"]["distance_threshold"])
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
    
    def _extract_from_split(self, timestamps: List[Path], 
                            config: dict[str, Any],
                            scene_object_finder: SceneObjectFinder,
                            scene_object_selector: SceneObjectSelector,
                            global_start_end_timestamps: Tuple[int, int]) -> List[Tuple[str, BoundingBox, str]]:
        traj = DefaultTrajectory(self.trajectory_dir, timestamps=timestamps)
    
        rgb_timestamps = traj.rgb_left.timestamps
        odom_timestamps = traj.odometry.timestamps
        odom_indexer = TimestampIndexer(odom_timestamps)
        traj_2d = traj.odometry.to_traj_2d_array()

        result = []

        for i in range(0, len(rgb_timestamps), config["global_traj_interval"]):
            rgb_ts = rgb_timestamps[i]
            odom_ts, odom_idx = odom_indexer.query(rgb_ts)

            footsteps = traj_2d[odom_idx:(odom_idx + config["local_traj_horizon"]):config["local_traj_interval"], :]
            footsteps = footsteps[~np.isnan(footsteps).any(axis=1)]
            if footsteps.shape[0] < config["min_local_traj_length"]:
                continue
            
            footsteps = to_relative_frame(footsteps)
            img = traj.rgb_left.at(rgb_ts)
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
    trajectories_root = luigi.Parameter(description="Path to the trajectories root directory")
    config = luigi.Parameter(description="Path to the config file")

    def requires(self):
        trajectories = self._read_list()
        for trajectory in trajectories:
            yield ProcessSingleTrajectory(
                trajectory_dir=str(trajectory),
                config=self.config
            )

    def _read_list(self) -> List[Path]:
        trajectories_root = Path(self.trajectories_root)
        return sorted([e for e in trajectories_root.iterdir() if e.is_dir()])


def main(trajectories_root: str,
         config: str,
         n_workers: int = 0,
         local: bool = False,
         scheduler_host: str = "localhost",
         scheduler_port: int = 8082):
    trajectories_root = Path(trajectories_root)
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
            trajectories_root=str(trajectories_root),
            config=str(config)
        )],
        **kwargs
    )


if __name__ == "__main__":
    fire.Fire(main)
