from typing import Protocol, Any, Dict, Optional, Type, List
from pathlib import Path
from egowalk_pipelines.misc.types import PathLike


class ExtractionChannelError(Exception):
    """
    Exception raised for errors during extraction channel reader/writer operations.
    """

    def __init__(self,
                 message: str,
                 cause: Optional[Exception] = None) -> None:
        """
        Initialize the exception with an optional message and cause.

        Args:
            message: A description of the error.
            cause: The original exception that caused this error.
        """
        super(ExtractionChannelError, self).__init__(message)
        self._message = message
        self._cause = cause

    def __str__(self) -> str:
        """
        Return a string representation of the error.
        """
        if self._cause is not None:
            return f"{self._message} (caused by: {type(self._cause).__name__}: {self._cause})"
        return self._message


class ExtractionChannelReader(Protocol):
    """
    A protocol for reading data from raw .svo/.svo2 EgoWalk recordings.

    Relies on the ZED SDK (pyzed).
    """

    def prepare_params(self,
                       init_params) -> None:
        """
        Prepare the initialization parameters for the ZED (virtual) camera.

        This method should be implemented if reader needs to populate channel-specific parameters.

        Args:
            init_params: sl.InitParameters The initialization parameters for the ZED camera. Variable is updated in place.
        """
        ...

    def setup(self,
              raw_recording_file_path: PathLike,
              zed) -> None:
        """
        Setup the ZED (virtual) camera and reader's internal state.

        This method should be implemented if reader needs to modify camera's state or initialize own internal state.

        Args:
            raw_recording_file_path: Path to the raw recording file being processed.
            zed: The ZED camera sl.Camera object. Updated in place.
        """
        ...

    def read(self,
             zed) -> Optional[Any]:
        """
        Read data from the ZED (virtual) camera.

        Args:
            zed: The ZED camera object with the current state.

        Returns:
            The data read from the ZED sl.Camera object (depends on the implementation).
        """
        ...

    def close(self) -> None:
        """
        Clean up the reader's internal state.
        """
        ...


class ExtractionChannelWriter(Protocol):
    """
    A protocol for writing data that was extracted from raw .svo/.svo2 EgoWalk 
    recordings using ExtractionChannelReader protocol.   
    """

    def open(self,
             raw_recording_file_path: PathLike,
             extraction_path: PathLike) -> None:
        """
        Open the channel for writing.

        This method is responsible for initializing the state, creating and opening the files, etc.

        Args:
            raw_recording_file_path: Path to the raw recording file being processed.
            extraction_path: Path to the extracted data.
        """
        ...

    def write(self,
              timestamp: int,
              data: Optional[Any]) -> None:
        """
        Write the data to the channel.

        Args:
            timestamp: Timestamp of the data.
            data: Data to write (depends on the implementation).
        """
        ...

    def close(self,
              success: bool) -> None:
        """
        Handles all files closing, state cleanup, etc.

        Args:
            success: Whether the extraction was successful.
        """
        ...

    def get_assets(self,
                   raw_recording_file_path: PathLike,
                   extraction_path: PathLike) -> List[Path]:
        """
        Get the assets for the channel.
        """
        ...


class ExtractionChannel:
    """
    Represents a data channel that is used during the extraction data from raw .svo/.svo2 EgoWalk recordings.

    During the extraction, multiple channels can be used to extract different types of data (e.g. images, odometry, depth, etc.)

    This class is used to unify the interface for the data extraction of different types.
    Each channel mainly contains the reader, responsible for reading data, and writer, which is using for saving the data in file system.
    """

    def __init__(self,
                 name: str,
                 reader_cls: Type[ExtractionChannelReader],
                 reader_kwargs: Dict[str, Any],
                 writer_cls: Type[ExtractionChannelWriter],
                 writer_kwargs: Dict[str, Any]):
        """
        Initialize the channel.

        Args:
            name: Name of the channel.
            reader_cls: Class of the reader.
            reader_kwargs: Keyword arguments for the reader.
            writer_cls: Class of the writer.
            writer_kwargs: Keyword arguments for the writer.
        """
        self._name = name
        self._reader_cls = reader_cls
        self._reader_kwargs = reader_kwargs
        self._writer_cls = writer_cls
        self._writer_kwargs = writer_kwargs

    @property
    def name(self) -> str:
        """
        Name of the channel.
        """
        return self._name

    def create_reader(self) -> ExtractionChannelReader:
        """
        Instantiates the reader with the keyword arguments passed during the channel initialization.

        Returns:
            The instantiated reader.
        """
        return self._reader_cls(**self._reader_kwargs)

    def create_writer(self) -> ExtractionChannelWriter:
        """
        Instantiates the writer with the keyword arguments passed during the channel initialization.

        Returns:
            The instantiated writer.
        """
        return self._writer_cls(**self._writer_kwargs)
