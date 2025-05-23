import autoroot
import yaml
import fire
import torch
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer,
                          BitsAndBytesConfig)
from egowalk_dataset.datasets.trajectory.trajectory import EgoWalkTrajectory
from egowalk_pipelines.utils.io_utils import SequentialCSVWriter


MODEL_PATH = "THUDM/cogvlm2-llama3-chat-19B"
TORCH_TYPE = torch.bfloat16
DEVICE = "cuda"
OUTPUT_FILE_NAME = "annotation_boxes_captions.csv"



def recur_move_to(item, tgt, criterion_func):
    if criterion_func(item):
        device_copy = item.to(tgt)
        return device_copy
    elif isinstance(item, list):
        return [recur_move_to(v, tgt, criterion_func) for v in item]
    elif isinstance(item, tuple):
        return tuple([recur_move_to(v, tgt, criterion_func) for v in item])
    elif isinstance(item, dict):
        return {k: recur_move_to(v, tgt, criterion_func) for k, v in item.items()}
    else:
        return item

def collate_fn(features, tokenizer) -> dict:
    images = [feature.pop('images', None) for feature in features if 'images' in feature]
    tokenizer.pad_token = tokenizer.eos_token
    max_length = max(len(feature['input_ids']) for feature in features)
    def pad_to_max_length(feature, max_length):
        padding_length = max_length - len(feature['input_ids'])
        feature['input_ids'] = torch.cat([feature['input_ids'], torch.full((padding_length,), tokenizer.pad_token_id)])
        feature['token_type_ids'] = torch.cat([feature['token_type_ids'], torch.zeros(padding_length, dtype=torch.long)])
        feature['attention_mask'] = torch.cat([feature['attention_mask'], torch.zeros(padding_length, dtype=torch.long)])
        if feature['labels'] is not None:
            feature['labels'] = torch.cat([feature['labels'], torch.full((padding_length,), tokenizer.pad_token_id)])
        else:
            feature['labels'] = torch.full((max_length,), tokenizer.pad_token_id)
        return feature
    features = [pad_to_max_length(feature, max_length) for feature in features]
    batch = {
        key: torch.stack([feature[key] for feature in features])
        for key in features[0].keys()
    }
    if images:
        batch['images'] = images
    return batch


def enlarge_box(box_x, box_y, box_w, box_h, image_width, image_height, padding=20):
    """
    Enlarge a bounding box by adding a fixed margin on all sides.
    
    Args:
        box_x, box_y: Top-left corner coordinates
        box_w, box_h: Width and height of the box
        image_width, image_height: Dimensions of the original image
        margin: Number of pixels to add on each side (default: 20)
        
    Returns:
        Tuple of (new_x, new_y, new_w, new_h)
    """
    # Calculate new box coordinates with margin
    new_x = max(0, box_x - padding)
    new_y = max(0, box_y - padding)
    
    # Calculate new width and height ensuring they don't exceed image boundaries
    new_w = min(image_width - new_x, box_w + 2 * padding)
    new_h = min(image_height - new_y, box_h + 2 * padding)
    
    return new_x, new_y, new_w, new_h


def store_captions(traj_path: Path,
                   captions_list: list[str],
                   boxes_df: pd.DataFrame):
    csv_writer = SequentialCSVWriter(columns=["timestamp", "box_x", "box_y", "box_w", "box_h", "caption"])
    for i, row in boxes_df.iterrows():
        csv_writer.add_line((row["timestamp"], 
                           row["box_x"], 
                           row["box_y"], 
                           row["box_w"], 
                           row["box_h"],
                           captions_list[i]))
    csv_writer.dump(traj_path / OUTPUT_FILE_NAME)


def process_single_trajectory(data_root: Path,
                              annotation_path: Path,
                               batch_size: int,
                               prompt: str,
                               box_padding: int,
                               tokenizer: AutoTokenizer,
                               model: AutoModelForCausalLM,
                               max_new_tokens: int):
    if (annotation_path / OUTPUT_FILE_NAME).is_file():
        return

    boxes_df = pd.read_csv(annotation_path / "annotation_boxes.csv")
    traj_rgb = EgoWalkTrajectory.from_dataset(annotation_path.name, data_root).rgb

    length = len(boxes_df)

    # Dictionary to store image names as keys and captions as values
    captions_list = []

    for idx in tqdm(range(0, length, batch_size), leave=False, desc=f"Processing {annotation_path.name}"):
        i_list = []
        for i in range(batch_size):
            if idx + i < length:
                i_list.append(boxes_df.iloc[idx + i])
            else:
                break

        input_sample_list = []
        for i in i_list:
            ts = int(i["timestamp"])
            box_x = int(i["box_x"])
            box_y = int(i["box_y"])
            box_w = int(i["box_w"])
            box_h = int(i["box_h"])
            box_x, box_y, box_w, box_h = enlarge_box(box_x, box_y, box_w, box_h, 960, 600, padding=box_padding)
            img = traj_rgb.at(ts)
            img = img[box_y:box_y+box_h, box_x:box_x+box_w]
            img = Image.fromarray(img)
            input_sample = model.build_conversation_input_ids(tokenizer, query=prompt, history=[], images=[img], template_version='chat')
            input_sample_list.append(input_sample)

        input_batch = collate_fn(input_sample_list, tokenizer)
        input_batch = recur_move_to(input_batch, DEVICE, lambda x: isinstance(x, torch.Tensor))
        input_batch = recur_move_to(input_batch, torch.bfloat16, lambda x: isinstance(x, torch.Tensor) and torch.is_floating_point(x))

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": 128002,
            "top_k": 1,
        }

        with torch.no_grad():
            outputs = model.generate(**input_batch, **gen_kwargs)
            outputs = outputs[:, input_batch['input_ids'].shape[1]:]
            outputs = tokenizer.batch_decode(outputs)

        outlist = [output.split("<|end_of_text|>")[0].strip() for output in outputs]

        # Add captions to the dictionary
        for i, caption in zip(i_list, outlist):
            captions_list.append(caption)

    store_captions(annotation_path, captions_list, boxes_df)


def main(data_root: str,
         annotation_root: str,
         config: str,
         batch_size: int = 3,
         hf_cache_dir: str | None = None):
    with open(config, 'r') as f:
        config = yaml.safe_load(f)
    prompt = config["prompt"]
    box_padding = config["box_padding"]
    max_new_tokens = config["max_new_tokens"]

    data_root = Path(data_root)
    annotation_root = Path(annotation_root)

    all_traj_dirs = list(annotation_root.glob("*/"))

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        cache_dir=hf_cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True,
        device_map=DEVICE,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        low_cpu_mem_usage=True,
        cache_dir=hf_cache_dir
    ).eval()

    for traj_path in tqdm(all_traj_dirs):
        process_single_trajectory(data_root=data_root,
                                  annotation_path=traj_path,
                                  batch_size=batch_size,
                                  prompt=prompt,
                                  box_padding=box_padding,
                                  tokenizer=tokenizer,
                                  model=model,
                                  max_new_tokens=max_new_tokens)


if __name__ == "__main__":
    fire.Fire(main)
