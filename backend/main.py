# backend/main.py
import os
import uuid
import json

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import cv2

from segmentation import get_segmentation_masks
from texture_apply import apply_texture_simple, apply_texture_floor_perspective

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
TILES_DIR = os.path.join(BASE_DIR, "tiles")
CATALOGUE_PATH = os.path.join(BASE_DIR, "catalogue.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(TILES_DIR, exist_ok=True)

# Load catalogue
if os.path.exists(CATALOGUE_PATH):
    with open(CATALOGUE_PATH, "r", encoding="utf-8") as f:
        CATALOGUE = json.load(f)
else:
    CATALOGUE = []

# Map id -> entry
CATALOGUE_MAP = {item["id"]: item for item in CATALOGUE}

app = FastAPI()

# Allow CORS from frontend (e.g. localhost:5500 or file://)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage of image metadata
ROOMS = {}  # image_id -> dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/catalog")
def get_catalog():
    """Return the list of available tiles."""
    return CATALOGUE


@app.post("/upload")
async def upload_room(image: UploadFile = File(...)):
    """Upload a room photo and run segmentation."""
    # Create unique image id
    image_id = str(uuid.uuid4())
    ext = os.path.splitext(image.filename)[1] or ".jpg"
    img_path = os.path.join(UPLOAD_DIR, f"{image_id}{ext}")

    # Save uploaded file
    with open(img_path, "wb") as f:
        f.write(await image.read())

    # Run segmentation
    try:
        wall_mask, floor_mask = get_segmentation_masks(img_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {e}")

    # Save masks
    wall_path = os.path.join(UPLOAD_DIR, f"{image_id}_wall.png")
    floor_path = os.path.join(UPLOAD_DIR, f"{image_id}_floor.png")
    cv2.imwrite(wall_path, wall_mask)
    cv2.imwrite(floor_path, floor_mask)

    # Save metadata
    ROOMS[image_id] = {
        "image_path": img_path,
        "wall_mask_path": wall_path,
        "floor_mask_path": floor_path,
    }

    return {"image_id": image_id}


@app.get("/apply_texture")
def apply_texture(image_id: str, region: str, texture_id: str):
    """
    Apply a texture to a given region ("wall" or "floor") and return the new image.
    """
    if image_id not in ROOMS:
        raise HTTPException(status_code=404, detail="image_id not found")

    if region not in ("wall", "floor"):
        raise HTTPException(status_code=400, detail="region must be 'wall' or 'floor'")

    if texture_id not in CATALOGUE_MAP:
        raise HTTPException(status_code=404, detail="texture_id not found in catalogue")

    room = ROOMS[image_id]
    img = cv2.imread(room["image_path"])
    if img is None:
        raise HTTPException(status_code=500, detail="Could not read original image")

    mask_path_key = f"{region}_mask_path"
    mask = cv2.imread(room[mask_path_key], cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise HTTPException(status_code=500, detail="Could not read mask image")

    texture_entry = CATALOGUE_MAP[texture_id]
    texture_path = os.path.join(TILES_DIR, texture_entry["file"])
    texture = cv2.imread(texture_path)
    if texture is None:
        raise HTTPException(status_code=500, detail="Could not read texture image")

    # Apply texture
    if region == "floor":
        result_img = apply_texture_floor_perspective(img, mask, texture)
    else:
        # wall or others use simple tiling
        result_img = apply_texture_simple(img, mask, texture)

    # Save result
    result_filename = f"{image_id}_{region}_{texture_id}.jpg"
    result_path = os.path.join(RESULT_DIR, result_filename)
    cv2.imwrite(result_path, result_img)

    return FileResponse(result_path, media_type="image/jpeg")
