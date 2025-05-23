import enum
import traceback
import json
import shutil
import time

try:
    import pyzed.sl as sl
except:
    print("Warning: ZED SDK (pyzed) is not installed, some functionality is unavailable")


from abc import ABC, abstractmethod
from typing import List, Optional, Union, Callable, Dict, Tuple
from pathlib import Path
from egowalk_pipelines.misc.types import PathLike
from egowalk_pipelines.extraction.channels import (ExtractionChannel,
                                               ExtractionChannelReader,
                                               ExtractionChannelWriter)
from egowalk_pipelines.utils.io_utils import is_svo_file
from egowalk_pipelines.utils.sync_utils import FPSBuffer
from egowalk_pipelines.utils.str_utils import seconds_to_human_readable_time
from egowalk_pipelines.misc.constants import METADATA_FILE_NAME, DEFAULT_FPS


PROGRESS_FLAG_FILE = "progress.flag"


class AbstractExtractionResult(ABC):
    """
    Abstract base class for the EgoWalk raw recordings extraction results.
    """

    def __init__(self,
                 input_svo_file: PathLike) -> None:
        self._input_svo_file = Path(input_svo_file)

    @property
    def input_svo_file(self) -> Path:
        """
        Path to the input SVO/SVO2 file.
        """
        return self._input_svo_file

    @abstractmethod
    def __str__(self) -> str:
        """
        String representation of the extraction result.
        """
        pass


class SuccessfulExtractionResult(AbstractExtractionResult):
    """
    Result of the successful EgoWalk raw recordings extraction.
    """

    def __init__(self,
                 input_svo_file: PathLike,
                 output_files: List[PathLike],
                 messages: Optional[List[str]] = None) -> None:
        """
        Initialize the successful extraction result.

        Args:
            input_svo_file: Path to the input SVO/SVO2 file.
            output_dir: Path to the output extracted data directory.
            messages: List of messages that were logged during the extraction.
        """
        super(SuccessfulExtractionResult, self).__init__(input_svo_file)
        self._output_files = [Path(e) for e in output_files]
        if messages is None:
            self._messages = []
        else:
            self._messages = [e for e in messages]

    @property
    def output_files(self) -> List[Path]:
        """
        List of paths to the output extracted data files.
        """
        return self._output_files

    @property
    def messages(self) -> List[str]:
        """
        List of messages that were logged during the extraction.
        """
        return self._messages

    def __str__(self) -> str:
        """
        String representation of the successful extraction result.
        """
        info_string = f"Successfully extracted {self._input_svo_file} to {self._output_files}"
        if len(self._messages) != 0:
            info_string = info_string + "\nMessages:"
            for message in self._messages:
                info_string = info_string + "\n" + message
        return info_string


class FailedExtractionResult(AbstractExtractionResult):
    """
    Result of the failed EgoWalk raw recordings extraction.
    """

    def __init__(self,
                 input_svo_file: PathLike,
                 message: str,
                 stack_trace: Optional[str] = None) -> None:
        """
        Initialize the failed extraction result.

        Args:
            input_svo_file: Path to the input SVO/SVO2 file.
            message: Error message.
            stack_trace: Stack trace of the error.
        """
        super(FailedExtractionResult, self).__init__(input_svo_file)
        self._message = message
        self._stack_trace = stack_trace

    @property
    def message(self) -> str:
        """
        Error message.
        """
        return self._message

    @property
    def stack_trace(self) -> Optional[str]:
        """
        Stack trace of the error.
        """
        return self._stack_trace

    def __str__(self) -> str:
        """
        String representation of the failed extraction result.
        """
        info_string = f"Extraction failed for {self._input_svo_file}\nError message: {self._message}"
        if self._stack_trace is not None:
            info_string = info_string + "\n" + self._stack_trace
        return info_string


class SkippedExtractionResult(AbstractExtractionResult):
    """
    Result of the skipped EgoWalk raw recordings extraction.
    """

    def __init__(self, input_svo_file: PathLike) -> None:
        """
        Initialize the skipped extraction result.
        """
        super(SkippedExtractionResult, self).__init__(input_svo_file)

    def __str__(self) -> str:
        """
        String representation of the skipped extraction result.
        """
        return f"Extraction skipped for {self._input_svo_file}"


class ExtractionOverwritePolicy(enum.Enum):
    """
    EgoWalk raw recordings extraction result overwrite policy.
    """

    ALL = "all"
    """
    Overwrite all existing extraction results.
    """

    INVALID = "invalid"
    """
    Overwrite only invalid extraction results.
    """

    NEVER = "never"
    """
    Never overwrite existing extraction results.
    """


class SVOExtractor:
    """
    Data extractor for the raw EgoWalk SVO/SVO2 recordings.

    This class will works according to the following scheme:
    1. Open the input .svo/.svo2 file
    2. Create and setup readers and writers for each specified channel
    3. Do required ZED SDK "virtual camera" setup
    4. Start obtaining data frame-by-frame using grab() method from ZED SDK
    5. Persist data using writers and close all resources

    This class will try not to throw any exceptions, using children of AbstractExtractionResult instead.
    However, anything may happen.
    """

    _KEY_RATE = "extraction_fps"

    def __init__(self,
                 channels: List[ExtractionChannel],
                 overwrite_policy: Union[ExtractionOverwritePolicy, str],
                 extraction_fps: Optional[int] = None,
                 progress_callback: Optional[Callable[[int, int], None]] = None,
                 verbose: bool = False,
                 zed_sdk_verbose: bool = False):
        """
        Initialize the SVO extractor.

        Args:
            channels: List of extraction channels.
            overwrite_policy: Overwrite policy.
            extraction_fps: Extraction rate.
            progress_callback: Callback function with args (current_frame, total_frames) to report the progress of the extraction.
            verbose: Verbose mode for the extractor messages.
            zed_sdk_verbose: Verbose mode for the ZED SDK.
        """
        self._channels = channels
        if not isinstance(overwrite_policy, ExtractionOverwritePolicy):
            overwrite_policy = ExtractionOverwritePolicy(overwrite_policy)
        self._overwrite_policy = overwrite_policy
        self._rate = int(
            extraction_fps) if extraction_fps is not None else None
        self._progress_callback = progress_callback
        self._verbose = verbose
        self._zed_sdk_verbose = int(zed_sdk_verbose)

    def __call__(self,
                 input_svo_file: PathLike,
                 output_parent_dir: PathLike) -> AbstractExtractionResult:
        """
        Extract data from the raw EgoWalk SVO/SVO2 recording.

        The extraction result is saved in the output_parent_dir/<input_svo_file_name without extension> directory.
        This directory will be passed to the extraction_validator.

        Args:
            input_svo_file: Path to the input SVO/SVO2 file.
            output_parent_dir: Path to the output parent directory.

        Returns:
            Extraction result (successful, failed or skipped).
        """
        try:
            return self._do_extract(input_svo_file, output_parent_dir)
        except Exception as e:
            return FailedExtractionResult(input_svo_file,
                                          str(e),
                                          traceback.format_exc())

    def _do_extract(self,
                    input_svo_file: PathLike,
                    output_parent_dir: PathLike) -> AbstractExtractionResult:
        input_svo_file = Path(input_svo_file)
        output_parent_dir = Path(output_parent_dir)

        # Input file validation
        if not is_svo_file(input_svo_file):
            return FailedExtractionResult(input_svo_file,
                                          f"Input file {input_svo_file} is not an SVO/SVO2 file")

        # Create readers and writers
        readers = {e.name: e.create_reader() for e in self._channels}
        writers = {e.name: e.create_writer() for e in self._channels}

        if self._skip_or_clean(input_svo_file, output_parent_dir, writers):
            return SkippedExtractionResult(input_svo_file)

        self._print(f"Starting processing {input_svo_file}")

        start_time = time.time()

        # Initialize ZED parameters
        init_params = sl.InitParameters(sdk_verbose=self._zed_sdk_verbose)
        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
        init_params.set_from_svo_file(str(input_svo_file))
        init_params.svo_real_time_mode = False
        # Specifically, update parameters using readers
        for reader in readers.values():
            reader.prepare_params(init_params)

        # Open "camera"
        zed = sl.Camera()
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            msg = f"Failed to open Zed object for {input_svo_file}, error code: {err}"
            self._print(msg)
            zed.close()
            return FailedExtractionResult(input_svo_file, msg)

        # Setup readers using opened "camera"
        result = self._setup_readers_safe(input_svo_file, zed, readers)
        if result is not None:
            self._close_all(zed, readers, None, False)
            return result

        # Setup writers using opened "camera"
        result = self._setup_writers_safe(input_svo_file,
                                          output_parent_dir,
                                          writers)
        if result is not None:
            self._close_all(zed, readers, writers, False)
            return result

        # Do the actual data reading in the grab loop
        result, n_frames = self._grab_loop(input_svo_file,
                                           zed,
                                           readers,
                                           writers)
        if result is not None:
            # Failed result is returned
            self._close_all(zed, readers, writers, False)
            time_elapsed, video_time = self._report_durations(
                start_time, n_frames)
            if video_time is not None:
                self._print(f"Failed {input_svo_file} in {time_elapsed}, "
                            f"estimated video time is {video_time}")
            else:
                self._print(f"Failed {input_svo_file} in {time_elapsed}")
            return result

        # Close everything and return result
        self._close_all(zed, readers, writers, True)

        time_elapsed, video_time = self._report_durations(start_time, n_frames)
        if video_time is not None:
            self._print(f"Finished {input_svo_file} in {time_elapsed}, "
                        f"estimated video time is {video_time}")
        else:
            self._print(f"Finished {input_svo_file} in {time_elapsed}")

        output_files = []
        for writer in writers.values():
            output_files.extend(writer.get_assets(input_svo_file, output_parent_dir))

        return SuccessfulExtractionResult(input_svo_file,
                                          output_files,
                                          [])

    def _grab_loop(self,
                   input_svo_file: Path,
                   zed,
                   readers: Dict[str, ExtractionChannelReader],
                   writers: Dict[str, ExtractionChannelWriter]) -> Tuple[FailedExtractionResult, int]:
        # Setup FPS buffer to reach target FPS rate
        fps_buffer = FPSBuffer(self._rate)

        frame_counter = 0
        grab_counter = 0
        total_frames = zed.get_svo_number_of_frames()
        rt_param = sl.RuntimeParameters()

        while True:
            # Main loop: uses zed.grab() standard method for grabbing current data frame
            err = zed.grab(rt_param)

            if err == sl.ERROR_CODE.SUCCESS:
                timestamp = zed.get_timestamp(
                    sl.TIME_REFERENCE.IMAGE).get_milliseconds()

                data = {}
                for name, reader in readers.items():
                    try:
                        data[name] = reader.read(zed)
                    except Exception as e:
                        msg = f"Failed to read channel '{name}' for {input_svo_file} with error:\n{str(e)}"
                        self._print(msg)
                        return FailedExtractionResult(input_svo_file,
                                                      msg,
                                                      traceback.format_exc()), frame_counter

                timestamp, data = fps_buffer.filter(timestamp, data)

                if data is not None:
                    for name, writer in writers.items():
                        try:
                            writer.write(timestamp, data[name])
                        except Exception as e:
                            msg = f"Failed to write channel '{name}' for {input_svo_file} with error:\n{str(e)}"
                            self._print(msg)
                            return FailedExtractionResult(input_svo_file,
                                                          msg,
                                                          traceback.format_exc()), frame_counter
                    frame_counter += 1
                
                grab_counter += 1
                if self._progress_callback is not None:
                    self._progress_callback(grab_counter, total_frames)

            elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED or frame_counter >= 12:
                # This just tells that we reached the end of the SVO file and most likely no errors occurred
                return None, frame_counter

            else:
                # Some unknown error occurred
                msg = f"grab() returned code {err} for {input_svo_file}"
                self._print(msg)
                return FailedExtractionResult(input_svo_file, msg), frame_counter

    def _skip_or_clean(self,
                       raw_file: Path,
                       output_root: Path,
                       writers: Dict[str, ExtractionChannelWriter]) -> bool:
        policy = self._overwrite_policy
        if policy == ExtractionOverwritePolicy.ALL:
            to_delete = True
        
        all_assets = []
        all_exist = True
        any_exist = False
        for writer in writers.values():
            assets = writer.get_assets(raw_file, output_root)
            all_assets.extend(assets)
            for asset in assets:
                if not asset.exists():
                    all_exist = False
                else:
                    any_exist = True

        # If overwrite policy is ALL, then delete all assets and not skip
        if policy == ExtractionOverwritePolicy.ALL:
            to_delete = True
            to_skip = False

        elif policy == ExtractionOverwritePolicy.INVALID:
            # If trajectory is valid, then just skip it
            if all_exist:
                to_skip = True
                to_delete = False
            # If trajectory is invalid, then delete all assets and not skip
            else:
                to_delete = True
                to_skip = False

        # If policy is NEVER, then check if something exists for extraction and skip if it does       
        elif policy == ExtractionOverwritePolicy.NEVER:
            to_delete = False
            to_skip = any_exist
        else:
            raise ValueError(f"Invalid overwrite policy: {policy}")

        if to_delete:
            for asset in all_assets:
                self._delete(asset)
        
        if to_skip:
            self._print(f"Skipping {'valid' if all_exist else 'invalid'} "
                        f"extracted trajectory {raw_file} according to the '{policy}' policy")

        return to_skip

    def _delete(self, path_to_delete: Path) -> None:
        if path_to_delete.is_file():
            path_to_delete.unlink()
        elif path_to_delete.is_dir():
            shutil.rmtree(str(path_to_delete))

    def _setup_readers_safe(self,
                            input_svo_file: Path,
                            zed,
                            readers: Dict[str, ExtractionChannelReader]) -> Optional[FailedExtractionResult]:
        for name, reader in readers.items():
            try:
                reader.setup(input_svo_file, zed)
            except Exception as e:
                msg = f"Failed to setup reader '{name}' for {input_svo_file} with error:\n{str(e)}"
                self._print(msg)
                return FailedExtractionResult(input_svo_file,
                                              msg,
                                              traceback.format_exc())
        return None

    def _setup_writers_safe(self,
                            input_svo_file: Path,
                            extraction_path: Path,
                            writers: Dict[str, ExtractionChannelWriter]) -> Optional[FailedExtractionResult]:
        for name, writer in writers.items():
            try:
                writer.open(input_svo_file, extraction_path)
            except Exception as e:
                msg = f"Failed to setup writer '{name}' for {input_svo_file} with error:\n{str(e)}"
                self._print(msg)
                return FailedExtractionResult(input_svo_file,
                                              msg,
                                              traceback.format_exc())
        return None

    def _close_all(self,
                   zed,
                   readers: Optional[Dict[str, ExtractionChannelReader]],
                   writers: Dict[str, ExtractionChannelWriter],
                   success: bool):
        zed.close()
        if readers is not None:
            for reader in readers.values():
                reader.close()
        if writers is not None:
            for writer in writers.values():
                writer.close(success)

    def _print(self,
               message: str):
        if self._verbose:
            print(message)

    def _report_durations(self,
                          start_time: float,
                          n_frames: Optional[int] = None) -> Tuple[str, str]:
        finish_time = time.time()
        time_elapsed = seconds_to_human_readable_time(finish_time - start_time)
        if n_frames is not None and n_frames > 0:
            video_time = seconds_to_human_readable_time(int(round(n_frames
                                                                  / (self._rate if self._rate is not None
                                                                      else DEFAULT_FPS))))
        else:
            video_time = None
        return time_elapsed, video_time
