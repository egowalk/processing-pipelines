from typing import Union, List
from pathlib import Path


_SVO_EXTENSION = "svo"
_SVO2_EXTENSION = "svo2"


def check_if_extracted_dir(input_path: Union[str, Path]) -> bool:
    input_path = Path(input_path)
    if not (input_path / "odometry.csv").is_file():
        return False
    if not (input_path / "frames").is_dir():
        return False
    return True


def locate_svo_files(input_path: Union[str, Path]) -> List[Path]:
    input_path = Path(input_path)

    if input_path.is_file() and \
            (input_path.name.endswith(_SVO_EXTENSION) or input_path.name.endswith(_SVO2_EXTENSION)):
        return [input_path]

    if input_path.is_dir():
        return [path for path in input_path.rglob(f"*.{_SVO_EXTENSION}")] \
            + [path for path in input_path.rglob(f"*.{_SVO2_EXTENSION}")]
            
    if input_path.is_file() and input_path.name.endswith("txt"):
        with open(str(input_path), "r") as f:
            paths = f.readlines()
        paths = [Path(e.rstrip()) for e in paths]
        return paths
    
    raise ValueError(f"Input path {input_path} is not and .svo/.svo2 file nor directory")


def locate_extractions(input_path: Union[str, Path]) -> List[Path]:
    input_path = Path(input_path)
    if check_if_extracted_dir(input_path):
        return [input_path]
    return [e for e in input_path.rglob("*/") if check_if_extracted_dir(e)]


def locate_day_dirs_struct(root_path: Union[str, Path]) -> List[Path]:
    root_path = Path(root_path)
    all_day_time_dirs = []
    day_dirs = sorted(root_path.glob("*/"))
    for day_dir in day_dirs:
        day_time_dirs = sorted(day_dir.glob("*/"))
        all_day_time_dirs = all_day_time_dirs + day_time_dirs
    return all_day_time_dirs
