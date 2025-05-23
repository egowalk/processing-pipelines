import numpy as np

from pathlib import Path
from egowalk_tools.trajectory import DefaultTrajectory
from canguro_processing_tools.misc.types import PathLike


def split_traj(traj_path: PathLike, distance_threshold: float = 3.) -> list[list[int]]:
    initial_traj = DefaultTrajectory(Path(traj_path))
    rgb_timestamps = initial_traj.rgb_left.timestamps
    odom_timestamps = set(initial_traj.odometry.timestamps)
    
    splits = []
    current_split = []
    last_pose = None
    for rgb_ts in rgb_timestamps:

        if rgb_ts not in odom_timestamps:
            # Missing odom - it's ok
            current_split.append(rgb_ts)
            continue

        new_pose = initial_traj.odometry.at(rgb_ts)
        new_pose = np.array([new_pose.position.x, new_pose.position.y])
        if last_pose is None:
            # Pose sequence is not initialized yet
            last_pose = new_pose
            current_split.append(rgb_ts)
            continue

        dist = np.linalg.norm(new_pose - last_pose)
        last_pose = new_pose
        if dist < distance_threshold:
            # Trajectory is not cut
            current_split.append(rgb_ts)
            continue
        
        # Trajectory is cut
        splits.append(current_split)
        current_split = [rgb_ts]

    splits.append(current_split)

    return splits
