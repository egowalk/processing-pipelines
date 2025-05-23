# EgoWalk Processing Pipeline

Data extraction and processing pipelines used to assemble [EgoWalk dataset](https://huggingface.co/EgoWalk).

## Prerequisites

First, symlink your raw data folder inside dir_links folder. Raw recordings can be found on [HuggingFace Datasets Hub](https://huggingface.co/datasets/EgoWalk/raw-recordings). Also link your output directory:
```shell
ln -s <raw dataset> dir_links/raw
ln -s <your output root> dir_links/processed
```

Obtain [RAM](https://github.com/xinyu1205/recognize-anything), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and [SAM](https://github.com/facebookresearch/segment-anything) weights and configs and place them in `weights/` directory. See [config](config/annotation_boxes.yaml) for required assets.


Data extraction requires ZED SDK, which we propose to use inside Docker. To build a basic Docker images, use Makefile command:
```shell
make build_docker
```

Finally, Docker container can be run:
```shell
make run_docker
```

Inside the container, install the dependencies of this code:
```shell
cd ~/code
pip3 install -r requirements.txt
```

## Data Extraction and Annotation

Most of the pipelines are based on [luigi](https://github.com/spotify/luigi) framework. Luigi server for monitoring can be run both inside and outside Docker:
```shell
luigid
```

All further scripts must be run from the repo's root.

### Extraction

Use scripts in following order to extract base data from the raw `.svo2` files:
1. [extract_raw.py](scripts/extraction/extract_raw.py)
2. [generate_meta.py](scripts/extraction/generate_meta.py)

### Language Annotation

Use scripts in following order to produce language annotations. Annotations are first stored in the intermediate `.csv` files which are then repacked to the final `.parquet` files.
1. [goal_boxes_pipeline.py](scripts/annotation/goal_boxes_pipeline.py)
2. [captioning_pipeline.py](scripts/annotation/captioning_pipeline.py)
3. [captions_filtering.py](scripts/annotation/captions_filtering.py)
4. [pack_captions.py](scripts/annotation/pack_captions.py)

### Traversability Segmentation

See the [extract_traverse.py](scripts/annotation/extract_traverse.py) script.