import autoroot
import fire
import luigi

from typing import List, Union, Optional
from pathlib import Path
from egowalk_pipelines.extraction.locators import locate_svo_files
from egowalk_pipelines.extraction.channels import ExtractionChannel
from egowalk_pipelines.extraction.extractor import SVOExtractor, FailedExtractionResult
from egowalk_pipelines.extraction.readers import (RGBChannelReader, DepthChannelReader, OdometryChannelReader)
from egowalk_pipelines.extraction.writers import (BlurringVideoRGBChannelWriter,
                                                         VideoDepthChannelWriter,
                                                         OdometryChannelWriter)
from egowalk_pipelines.misc.constants import (DATA_DIR, VIDEO_DIR, RGB_DIR, DEPTH_DIR, ODOMETRY_DIR)


_HOST_DOCKER_HOST = "docker_host"

_EXTRACTION_FPS = 5


class ProcessSingleRecording(luigi.Task):
    input_file = luigi.Parameter(description="Path to the SVO file to process")
    output_root = luigi.Parameter(description="Path to the output root directory")
    batch_size = luigi.Parameter(description="Batch size for the face detection model")

    def output(self):
        input_file = Path(self.input_file)
        output_root_dir = Path(self.output_root)
        base_name = input_file.stem

        result_odometry = output_root_dir / DATA_DIR / f"{base_name}.parquet"
        # TODO: Check how to add rest of the files to the local target
        # result_rgb = output_root_dir / VIDEO_DIR / RGB_DIR / f"{base_name}__rgb.mp4"
        # result_depth = output_root_dir / VIDEO_DIR / DEPTH_DIR / f"{base_name}__depth.mkv"

        return luigi.LocalTarget(str(result_odometry))

    def run(self):
        print(f"Processing {self.input_file}...")
        extractor = self._create_extractor()
        input_file = Path(self.input_file)
        output_dir = Path(self.output_root)
        result = extractor(input_file, output_dir)
        if isinstance(result, FailedExtractionResult):
            raise RuntimeError(result.message)

    def _create_extractor(self):
        extractor = SVOExtractor(
            channels=[
                ExtractionChannel(
                    name="rgb",
                    reader_cls=RGBChannelReader,
                    reader_kwargs={"view": "left"},
                    writer_cls=BlurringVideoRGBChannelWriter,
                    writer_kwargs={"view": "left",
                                   "model_batch_size": int(self.batch_size)}
                ),
                ExtractionChannel(
                    name="depth",
                    reader_cls=DepthChannelReader,
                    reader_kwargs={"mode": "neural"},
                    writer_cls=VideoDepthChannelWriter,
                    writer_kwargs={}
                ),
                ExtractionChannel(
                    name="odometry",
                    reader_cls=OdometryChannelReader,
                    reader_kwargs={"gen_2_enabled": True, "verbose": True},
                    writer_cls=OdometryChannelWriter,
                    writer_kwargs={}
                )
            ],
            overwrite_policy="invalid",
            extraction_fps=_EXTRACTION_FPS,
            progress_callback=self._progress_monitor,
            verbose=True,
            zed_sdk_verbose=True
        )
        return extractor
    
    def _progress_monitor(self, 
                          current_frame: int, 
                          total_frames: int):
        if current_frame == total_frames:
            self.set_progress_percentage(100)
            return
        if current_frame % 300 == 0:
            self.set_progress_percentage(round((current_frame / total_frames) * 100, 2))


class ProcessAllRecordings(luigi.WrapperTask):
    raw_root = luigi.Parameter(description="Path to the raw root directory")
    files_list = luigi.Parameter(description="Path to the files list", default=None)
    output_root = luigi.Parameter(description="Path to the output root directory")
    batch_size = luigi.Parameter(description="Batch size for the face detection model")

    def requires(self):
        files = self._read_files_list()
        for file_path in files:
            yield ProcessSingleRecording(
                input_file=str(file_path),
                output_root=self.output_root,
                batch_size=self.batch_size
            )

    def _read_files_list(self) -> List[Path]:
        raw_root = Path(self.raw_root)
        if self.files_list is None:
            return locate_svo_files(raw_root)
        else:
            with open(self.files_list, "r") as f:
                paths = f.readlines()
            file_names = [e.rstrip() for e in paths]
            full_paths = [raw_root / e.split("__")[0] / e.split(".")[0] / e for e in file_names]
            return full_paths
  

def main(raw_root: str,
         output_root: str,
         batch_size: int,
         files_list: Optional[str] = None,
         n_workers: int = 0,
         local: bool = False,
         scheduler_host: str = "localhost",
         scheduler_port: int = 8082):
    raw_root = Path(raw_root)
    output_root = Path(output_root)

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root {raw_root} does not exist")
    if files_list is not None and not Path(files_list).exists():
        raise FileNotFoundError(f"Files list {files_list} does not exist")
    output_root.mkdir(parents=True, exist_ok=True)

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
        ProcessAllRecordings(
            raw_root=str(raw_root),
            files_list=files_list,
            output_root=str(output_root),
            batch_size=batch_size
        )],
        **kwargs
    )


if __name__ == "__main__":
    fire.Fire(main)
