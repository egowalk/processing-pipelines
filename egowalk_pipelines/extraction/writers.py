import sqlite3
import json
import av.container
import av.video
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import av
import cv2

from typing import Any, Optional, Dict, Tuple, Union, List
from pathlib import Path
from collections import deque
from fractions import Fraction
from egowalk_pipelines.misc.types import PathLike, RotTrans
from egowalk_pipelines.misc.constants import (
                                          VIEW_LEFT,
                                          VIEW_RIGHT,
                                          VIEW_BOTH)
from egowalk_pipelines.models.owl_vit import OWLVitFaceDetector
from egowalk_pipelines.utils.face_utils import FaceBlurrer


class DummyChannelWriter:
    """
    Dummy channel writer that does nothing.

    Follows the protocol of the ExtractionChannelWriter.
    """

    def open(self,
             raw_recording_file_path: PathLike,
             extraction_path: PathLike) -> None:
        pass

    def write(self,
              timestamp: int,
              data: Optional[Any]) -> None:
        pass

    def close(self,
              success: bool) -> None:
        pass

    def open(self,
             raw_recording_file_path: PathLike,
             extraction_path: PathLike) -> List[Path]:
        pass


class VideoWriter:

    def __init__(self,
                 codec: str,
                 codec_options: Dict[str, Any],
                 rate: int,
                 wh: Tuple[int, int],
                 pixel_format: str,
                 frame_format: str,
                 time_base: Optional[Fraction] = None):
        self._codec = codec
        self._codec_options = codec_options
        self._rate = rate
        self._wh = wh
        self._pixel_format = pixel_format
        self._frame_format = frame_format
        self._time_base = time_base

        self._counter = 0
        self._output_file: Optional[Path] = None
        self._temp_file: Optional[Path] = None
        self._container: Optional[av.container.OutputContainer] = None
        self._stream: Optional[av.video.stream.VideoStream] = None

    def open(self, output_file: Path, temp_file: Path):
        self._output_file = output_file
        self._temp_file = temp_file

        container = av.open(str(self._temp_file), "w", options={'movflags': 'faststart'})
        width, height = self._wh
        stream = container.add_stream(self._codec, rate=self._rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = self._pixel_format
        if self._time_base is not None:
            stream.time_base = self._time_base
        stream.options = self._codec_options
        self._container = container
        self._stream = stream
        self._counter = 0

    def write_frame(self, frame: np.ndarray):
        frame = av.VideoFrame.from_ndarray(frame, format=self._frame_format)
        frame.pts = self._counter
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self._counter += 1

    def close(self, keep_tmp: bool = False):
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        
        if not keep_tmp:
            self._temp_file.rename(self._output_file)
        self._container = None
        self._stream = None
        self._output_file = None
        self._temp_file = None
        self._counter = 0


class BlurringVideoRGBChannelWriter:

    _VIDEO_DIR = "video"
    _RGB_DIR = "rgb"

    _VIDEO_EXTENSION = "mp4"
    _TIMESTAMPS_EXTENSION = "npy"

    def __init__(self,
                 view: str,
                 model_batch_size: int,
                 model_device: str = "cuda"):
        assert view in (VIEW_LEFT,
                        VIEW_RIGHT), f"Wrong view {view}, possible values are {VIEW_LEFT}, {VIEW_RIGHT}"
        self._view = view
        self._face_detector = OWLVitFaceDetector(device=model_device)
        self._face_blurrer = FaceBlurrer()
        self._model_batch_size = model_batch_size

        self._tmp_file: Optional[Path] = None
        self._output_file: Optional[Path] = None
        self._deque: deque = None
        self._video_writer: Optional[VideoWriter] = None

    def open(self,
             raw_recording_file_path: PathLike,
             extraction_path: PathLike) -> None:
        raw_recording_file_path = Path(raw_recording_file_path)
        extraction_path = Path(extraction_path) / self._VIDEO_DIR / self._RGB_DIR
        if not extraction_path.exists():
            extraction_path.mkdir(parents=True, exist_ok=True)

        self._deque = deque(maxlen=self._model_batch_size)

        traj_name = raw_recording_file_path.stem
        self._output_file = extraction_path / f"{traj_name}__rgb.{self._VIDEO_EXTENSION}"
        self._tmp_file = extraction_path / f"{traj_name}__rgb__tmp.{self._VIDEO_EXTENSION}"

    def write(self,
              timestamp: int,
              data: Tuple[Optional[np.ndarray], Optional[np.ndarray]]) -> None:
        if self._view == VIEW_LEFT:
            frame, _ = data
        elif self._view == VIEW_RIGHT:
            _, frame = data
        else:
            raise ValueError(f"Wrong view {self._view}, possible values are {VIEW_LEFT}, {VIEW_RIGHT}")
        if frame is None:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        
        if self._video_writer is None:
            height, width = frame.shape[:2]
            self._video_writer = VideoWriter(codec="libx264",
                                             codec_options={
                                                'crf': '18',        # Constant Rate Factor - quality control
                                                'preset': 'ultrafast',     # Encoding speed/compression trade-off
                                                'x264-params': 'keyint=1:min-keyint=1:scenecut=0:ref=1:bframes=0:slices=16:trellis=0:'
                                                        'deblock=0:analyse=0x1:0x111:me=dia:subme=1:no-mbtree=1:fast_pskip=0:'
                                                        'no-mixed-refs=1:aq-mode=0',
                                                'tune': 'zerolatency'   # Tune for low latency access
                                             },
                                             rate=100,
                                             wh=(width, height),
                                             pixel_format="yuv444p",
                                             frame_format="rgb24",
                                             time_base=None)
            self._video_writer.open(self._output_file, self._tmp_file)
        
        self._deque.append(frame)
        
        if len(self._deque) == self._model_batch_size:
            self._process_batch()

    def close(self,
              success: bool) -> None:
        self._process_batch()
        self._video_writer.close(keep_tmp=not success)
        
        self._tmp_file = None
        self._output_file = None
        self._deque = None
        self._video_writer = None

    def get_assets(self,
                   raw_recording_file_path: PathLike,
                   extraction_path: PathLike) -> List[Path]:
        raw_recording_file_path = Path(raw_recording_file_path)
        extraction_path = Path(extraction_path) / self._VIDEO_DIR / self._RGB_DIR
        traj_name = raw_recording_file_path.stem
        output_file = extraction_path / f"{traj_name}__rgb.{self._VIDEO_EXTENSION}"
        return [output_file]

    def _process_batch(self):
        if len(self._deque) > 0:
            detected_faces = self._face_detector(list(self._deque))
            frames = [self._face_blurrer(frame, boxes) for frame, boxes in zip(self._deque, detected_faces)]
            for frame in frames:
                self._video_writer.write_frame(frame)
            self._deque.clear()


class VideoDepthChannelWriter:

    _VIDEO_EXTENSION = "mkv"
    _TIMESTAMPS_EXTENSION = "npy"

    _VIDEO_DIR = "video"
    _DEPTH_DIR = "depth"

    def __init__(self):
        self._tmp_file: Optional[Path] = None
        self._output_file: Optional[Path] = None
        self._video_writer: Optional[VideoWriter] = None

    def open(self,
             raw_recording_file_path: PathLike,
             extraction_path: PathLike) -> None:
        raw_recording_file_path = Path(raw_recording_file_path)
        extraction_path = Path(extraction_path) / self._VIDEO_DIR / self._DEPTH_DIR
        if not extraction_path.exists():
            extraction_path.mkdir(parents=True, exist_ok=True)

        traj_name = raw_recording_file_path.stem
        self._output_file = extraction_path / f"{traj_name}__depth.{self._VIDEO_EXTENSION}"
        self._tmp_file = extraction_path / f"{traj_name}__depth__tmp.{self._VIDEO_EXTENSION}"

    def write(self,
              timestamp: int,
              data: Optional[np.ndarray]) -> None:
        if data is None:
            return
        frame = data
    
        if self._video_writer is None:
            height, width = frame.shape
            self._video_writer = VideoWriter(codec="ffv1",
                                             codec_options={
                                                'level': '3',
                                                'slices': '16',  # Multiple slices can help with parallel decoding
                                                'slicecrc': '1',  # Add CRC check to each slice
                                                'g': '1'  # All frames are keyframes for best random access
                                             },
                                             rate=100,
                                             wh=(width, height),
                                             pixel_format="gray16le",
                                             frame_format="gray16le",
                                             time_base=None)
            self._video_writer.open(self._output_file, self._tmp_file)
                
        frame[np.isnan(frame)] = 0.
        frame[np.isinf(frame)] = 0.
        frame = frame * 1000.
        frame = np.round(frame).astype(np.uint16)
        self._video_writer.write_frame(frame)

    def close(self,
              success: bool) -> None:
        self._video_writer.close(keep_tmp=not success)
        
        self._tmp_file = None
        self._output_file = None
        self._video_writer = None

    def get_assets(self,
                   raw_recording_file_path: PathLike,
                   extraction_path: PathLike) -> List[Path]:
        raw_recording_file_path = Path(raw_recording_file_path)
        extraction_path = Path(extraction_path) / self._VIDEO_DIR / self._DEPTH_DIR
        traj_name = raw_recording_file_path.stem
        output_file = extraction_path / f"{traj_name}__depth.{self._VIDEO_EXTENSION}"
        return [output_file]


class OdometryChannelWriter:

    _DATA_DIR = "data"
    _DF_EXTENSION = "parquet"

    def __init__(self):
        self._output_file: Optional[Path] = None
        self._traj_name: Optional[str] = None
        self._timestamps = []
        self._indices = []
        self._names = []
        self._cart_x = []
        self._cart_y = []
        self._cart_z = []
        self._quat_x = []
        self._quat_y = []
        self._quat_z = []
        self._quat_w = []
        self._idx_counter = 0

    def open(self,
             raw_recording_file_path: PathLike,
             extraction_path: PathLike) -> None:
        raw_recording_file_path = Path(raw_recording_file_path)
        extraction_path = Path(extraction_path) / OdometryChannelWriter._DATA_DIR
        if not extraction_path.exists():
            extraction_path.mkdir(parents=True, exist_ok=True)

        self._traj_name = raw_recording_file_path.stem
        self._output_file = extraction_path / f"{self._traj_name}.{OdometryChannelWriter._DF_EXTENSION}"

    def write(self,
              timestamp: int,
              data: Optional[RotTrans]) -> None:
        if data is not None:
            quat_x, quat_y, quat_z, quat_w, cart_x, cart_y, cart_z = data
        else:
            quat_x, quat_y, quat_z, quat_w, cart_x, cart_y, cart_z = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        self._timestamps.append(timestamp)
        self._indices.append(self._idx_counter)
        self._names.append(self._traj_name)
        self._cart_x.append(cart_x)
        self._cart_y.append(cart_y)
        self._cart_z.append(cart_z)
        self._quat_x.append(quat_x)
        self._quat_y.append(quat_y)
        self._quat_z.append(quat_z)
        self._quat_w.append(quat_w)
        self._idx_counter += 1

    def close(self,
              success: bool) -> None:
        if success:
            dataset_dict = {
                "timestamp": self._timestamps,
                "trajectory": self._names,
                "frame": self._indices,
                "cart_x": self._cart_x,
                "cart_y": self._cart_y,
                "cart_z": self._cart_z,
                "quat_x": self._quat_x,
                "quat_y": self._quat_y,
                "quat_z": self._quat_z,
                "quat_w": self._quat_w,
            }
            df = pd.DataFrame(dataset_dict)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, str(self._output_file))
        
        self._output_file = None
        self._traj_name = None
        self._timestamps = []
        self._indices = []
        self._names = []
        self._cart_x = []
        self._cart_y = []
        self._cart_z = []
        self._quat_x = []
        self._quat_y = []
        self._quat_z = []
        self._quat_w = []
        self._idx_counter = 0

    def get_assets(self,
                   raw_recording_file_path: PathLike,
                   extraction_path: PathLike) -> List[Path]:
        raw_recording_file_path = Path(raw_recording_file_path)
        extraction_path = Path(extraction_path) / self._DATA_DIR
        traj_name = raw_recording_file_path.stem
        output_file = extraction_path / f"{traj_name}.{self._DF_EXTENSION}"
        return [output_file]
