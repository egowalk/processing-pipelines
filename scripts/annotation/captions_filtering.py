import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from egowalk_dataset.datasets.trajectory.trajectory import EgoWalkTrajectory
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="caption_filtering_normal_v2.log",
)

# Dataset setup
TRAJ_ROOT = "/home/captain/data/processed/annotation_temp"
ALL_TRAJ_DIRS = list(Path(TRAJ_ROOT).glob("*/"))
total_good_captions = 0
total_skipped_captions = 0

# Gemma-3 model setup
model_id = "google/gemma-3-27b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="cuda"
).eval()
processor = AutoProcessor.from_pretrained(model_id)


SYSTEM_PROMPT = """
You are given captions that were extracted for the various image patches. Those patches are potential goals for a robot to navigate to. We need to do a filtration of those captions using the following criteria:

The confidence of the caption: If the caption fails to identify the specific object and instead relies on abstract terms such as "object," "structure," "thing," or "surface," it should be rejected. Additionally, captions that contain excessive expressions of uncertainty, including words like "probably," "likely," or "appears," should also be rejected.

Along with the filtration, you need to reformulate the captions that are not rejected. Those captions should be become very brief and do not include introductory words like "The image shows...", "The main object is....", etc. Keep all the main properties of the caption; just remove unnecessary words and compress the information.

For the captions that you reject, output single word "REJECT". For the captions that you keep, output the reformulated version.

Input:  """


def evaluate_caption(caption):
    """Evaluate a single caption using Gemma-3."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": caption}]},
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=60, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True).strip()
    return decoded

# Process each trajectory
for traj_dir in ALL_TRAJ_DIRS:
    logging.info(f"Processing trajectory: {traj_dir}")
    captions_df = pd.read_csv(traj_dir / "annotation_boxes_captions.csv")

    # List to store good captions
    good_captions = []
    bad_captions = []
    skipped_captions = 0

    # Evaluate each caption
    for idx, row in captions_df.iterrows():
        caption = row["caption"]
        result = evaluate_caption(caption)
        
        timestamp = row["timestamp"]
        box_x = row["box_x"]
        box_y = row["box_y"]
        box_w = row["box_w"]
        box_h = row["box_h"]
        if  "REJECT" in result:  # "Keep":
            skipped_captions += 1
            bad_captions.append({
                "timestamp": timestamp,
                "box_x": box_x,
                "box_y": box_y,
                "box_w": box_w,
                "box_h": box_h,
                "caption": caption,
            })

        else:
            good_captions.append({
                "timestamp": timestamp,
                "box_x": box_x,
                "box_y": box_y,
                "box_w": box_w,
                "box_h": box_h,
                "caption": result,  # Use the reformulated caption
            })

    # Save good captions to a new CSV in the same trajectory folder
    if good_captions:
        filtered_df = pd.DataFrame(good_captions)
        output_csv = (
            traj_dir / "filtered_captions_normal_v2.csv"
        )  # Save in traj_dir, not a separate folder
        filtered_df.to_csv(output_csv, index=False)
        logging.info(
            f"Saved {len(good_captions)} good captions to {output_csv}, skipped {skipped_captions} captions"
        )
        total_good_captions += len(good_captions)
        total_skipped_captions += skipped_captions
        logging.info(f"So far the total number of good captions: {total_good_captions}")
        logging.info(
            f"So far the total number of skipped captions: {total_skipped_captions}"
        )

    else:
        logging.info(
            f"No good captions found for {traj_dir.name}, skipped {skipped_captions} captions"
        )
        total_skipped_captions += skipped_captions

    if bad_captions:
        bad_df = pd.DataFrame(bad_captions)
        bad_output_csv = (
            traj_dir / "bad_captions_normal_v2.csv"
        )  # Save in traj_dir, not a separate folder
        bad_df.to_csv(bad_output_csv, index=False)
        logging.info(
            f"Saved {len(bad_captions)} bad captions to {bad_output_csv}, skipped {skipped_captions} captions"
        )

logging.info(f"Total number of good captions: {total_good_captions}")
logging.info(f"Total number of skipped captions: {total_skipped_captions}")
logging.info("Processing complete!")
