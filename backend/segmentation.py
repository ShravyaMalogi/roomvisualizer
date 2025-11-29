# backend/segmentation.py

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import cv2

"""
Segmentation for room visualizer:
- Use SegFormer (ADE20K) to find walls.
- Use a robust geometric heuristic for floors:
  -> always treat bottom ~30-35% as floor area.
"""

MODEL_ID = "nvidia/segformer-b3-finetuned-ade-512-512"

print(f"[segmentation] Loading model: {MODEL_ID} (first time may take a while)...")
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_ID)
model.eval()
print("[segmentation] Model loaded.")

id2label = model.config.id2label

# Keywords to detect walls from model labels
WALL_KEYWORDS = ("wall", "building", "structure")

WALL_IDS = []
for idx, label in id2label.items():
    lname = label.lower()
    if any(k in lname for k in WALL_KEYWORDS):
        WALL_IDS.append(idx)

print(f"[segmentation] Wall class IDs: {WALL_IDS}")


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected region in the mask."""
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask  # nothing or just background

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # skip background
    largest_mask = (labels == largest_label).astype(np.uint8) * 255
    return largest_mask.astype(np.uint8)


def _postprocess_wall(prediction: np.ndarray):
    """
    From full prediction (H,W) and WALL_IDS, return a clean wall mask.
    """
    if not WALL_IDS:
        return np.zeros_like(prediction, dtype=np.uint8)

    mask = np.isin(prediction, WALL_IDS).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    # Close small holes, then open to remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep only main wall region
    mask = _keep_largest_component(mask)
    return mask.astype(np.uint8)


def get_segmentation_masks(image_path: str):
    """
    Returns:
        wall_mask, floor_mask  as uint8 (0 or 255)
    Strategy:
        - Wall: from SegFormer + cleanup.
        - Floor: simple geometric heuristic = bottom 30-35% of image.
                 This is MUCH more reliable for tiling.
    """
    # Load image with PIL
    image = Image.open(image_path).convert("RGB")
    width, height = image.size  # (W, H)

    # --- WALL via model ---
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # [1, num_labels, h, w]

    # Upsample back to original image size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(height, width),  # (H, W)
        mode="bilinear",
        align_corners=False,
    )

    # Argmax over classes -> [H, W]
    pred = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.int32)

    wall_mask = _postprocess_wall(pred)

    # --- FLOOR via heuristic: bottom band of image ---
    floor_mask = np.zeros((height, width), dtype=np.uint8)
    # You can tweak this split (0.65 = 65% height for wall, rest floor)
    split_y = int(height * 0.65)
    floor_mask[split_y:, :] = 255

    return wall_mask.astype(np.uint8), floor_mask.astype(np.uint8)
