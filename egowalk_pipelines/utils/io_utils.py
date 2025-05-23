import csv

from typing import Tuple, Union
from pathlib import Path
from egowalk_pipelines.misc.types import PathLike
from egowalk_pipelines.misc.constants import EXTENSION_SVO, EXTENSION_SVO2


def is_svo_file(file_path: PathLike) -> bool:
    """
    Check if the file is an SVO/SVO2 file.

    file_path is considered as an SVO/SVO2 file if it exists and is a 
    file with one of the following extensions: .svo, .svo2.

    Args:
        file_path: Path to the file to check.

    Returns:
        True if the file is an SVO/SVO2 file, False otherwise.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return False
    if not file_path.is_file():
        return False
    if file_path.suffix not in [f".{EXTENSION_SVO}", 
                                f".{EXTENSION_SVO2}"]:
        return False
    return True



class SequentialCSVWriter:

    def __init__(self, columns: Tuple[str]) -> None:
        self._columns = columns
        self._lines = [self._columns]

    def add_line(self, row: Tuple[str]) -> None:
        self._lines.append(row)

    def dump(self, csv_file: Union[str, Path], clear: bool = True) -> None:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            for line in self._lines:
                writer.writerow(line)
        if clear:
            self._lines = [self._columns]
