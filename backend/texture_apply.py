# backend/texture_apply.py
import cv2
import numpy as np

def apply_texture_simple(orig_img, mask, texture_img):
    """
    Simple wall texturing:
    - Find bounding box of the mask.
    - Tile the texture only inside that box.
    - Blend back into the original image.
    """
    h, w = orig_img.shape[:2]

    # Ensure mask is single-channel uint8
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask.copy()

    # If mask is empty, just return original
    if cv2.countNonZero(mask_gray) == 0:
        return orig_img.copy()

    # Bounding box of the mask
    ys, xs = np.where(mask_gray > 0)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    roi_h = y_max - y_min + 1
    roi_w = x_max - x_min + 1

    # Create tiled texture for ROI
    tex_h, tex_w = texture_img.shape[:2]

    # Optional: scale texture to be a bit smaller for walls (more tiles visible)
    scale_factor = 0.5  # lower = smaller tiles
    new_tex_w = max(1, int(tex_w * scale_factor))
    new_tex_h = max(1, int(tex_h * scale_factor))
    tex_scaled = cv2.resize(texture_img, (new_tex_w, new_tex_h), interpolation=cv2.INTER_LINEAR)

    tex_h, tex_w = tex_scaled.shape[:2]
    rep_y = roi_h // tex_h + 2
    rep_x = roi_w // tex_w + 2
    tiled_roi = np.tile(tex_scaled, (rep_y, rep_x, 1))[:roi_h, :roi_w, :]

    # Prepare output
    output = orig_img.copy().astype(np.float32)

    # Extract ROI mask
    roi_mask = mask_gray[y_min:y_max+1, x_min:x_max+1]
    roi_mask_float = (roi_mask / 255.0).astype(np.float32)
    roi_mask_3 = cv2.merge([roi_mask_float, roi_mask_float, roi_mask_float])

    roi_orig = output[y_min:y_max+1, x_min:x_max+1, :]

    blended_roi = roi_orig * (1.0 - roi_mask_3) + tiled_roi.astype(np.float32) * roi_mask_3

    output[y_min:y_max+1, x_min:x_max+1, :] = blended_roi

    return np.clip(output, 0, 255).astype(np.uint8)


def apply_texture_floor_perspective(orig_img, floor_mask, texture_img):
    """
    Floor texturing with a fixed geometric floor:
    - Ignores the incoming floor_mask for geometry.
    - Treats the bottom ~35-40% of the image as floor.
    - Builds a simple trapezoid and warps a tiled texture into it.
    This is a "brute force" but very reliable Phase-1 approach.
    """
    h, w = orig_img.shape[:2]

    # 1) Define a simple floor band at the bottom of the image
    y_top = int(h * 0.6)   # start of floor (60% down)
    y_bottom = h - 1       # bottom of image

    # Trapezoid narrower at the top, full-width at the bottom
    top_margin = int(0.18 * w)  # how much to "pinch" at top
    x_left_top = top_margin
    x_right_top = w - top_margin

    quad = np.array([
        [x_left_top, y_top],      # top-left
        [x_right_top, y_top],     # top-right
        [w - 1,       y_bottom],  # bottom-right
        [0,           y_bottom],  # bottom-left
    ], dtype=np.float32)

    # 2) Create a synthetic floor mask that matches this band
    floor_mask_band = np.zeros((h, w), dtype=np.uint8)
    floor_mask_band[y_top:y_bottom+1, :] = 255

    # 3) Build a big tiled texture plane (source)
    tex_h, tex_w = texture_img.shape[:2]
    plane_h = h
    plane_w = w * 2  # wider so perspective has margin

    rep_y = plane_h // tex_h + 2
    rep_x = plane_w // tex_w + 2
    tiled_plane = np.tile(texture_img, (rep_y, rep_x, 1))[:plane_h, :plane_w, :]

    # 4) Perspective transform from full plane to our trapezoid
    src_pts = np.array([
        [0, 0],
        [plane_w - 1, 0],
        [plane_w - 1, plane_h - 1],
        [0, plane_h - 1]
    ], dtype=np.float32)

    dst_pts = quad.astype(np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(tiled_plane, M, (w, h))

    # 5) Blend only inside our floor mask band
    mask_gray = floor_mask_band
    mask_float = (mask_gray / 255.0).astype(np.float32)
    mask_3 = cv2.merge([mask_float, mask_float, mask_float])

    orig_f = orig_img.astype(np.float32)
    warped_f = warped.astype(np.float32)

    blended = orig_f * (1.0 - mask_3) + warped_f * mask_3
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended
