#!/usr/bin/env python3
import os
import io
import time
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import cairosvg
import moondream as md


# ──────────────────────────────────────────────────────────────────────────────
# SVG Rasterization
# ──────────────────────────────────────────────────────────────────────────────

def rasterize_mask(svg_path_str, bbox, image_size):
    img_w, img_h = image_size
    x_min = bbox["x_min"]
    y_min = bbox["y_min"]
    bw    = bbox["x_max"] - bbox["x_min"]
    bh    = bbox["y_max"] - bbox["y_min"]

    tx = x_min * img_w
    ty = y_min * img_h
    sx = bw * img_w
    sy = bh * img_h

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{img_w}" height="{img_h}" '
        f'viewBox="0 0 {img_w} {img_h}">'
        f'<g transform="translate({tx},{ty}) scale({sx},{sy})">'
        f'<path d="{svg_path_str}" fill="white"/>'
        f'</g>'
        f'</svg>'
    )

    png_bytes = cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        output_width=img_w,
        output_height=img_h,
    )
    pil_mask = Image.open(io.BytesIO(png_bytes)).convert("L")
    mask = np.array(pil_mask)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def union_segments(segments, image_size):
    combined = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    for seg in segments:
        try:
            mask = rasterize_mask(seg["path"], seg["bbox"], image_size)
            combined = cv2.bitwise_or(combined, mask)
        except Exception as e:
            print(f"        [WARN] rasterize failed: {e}")
    return combined


# ──────────────────────────────────────────────────────────────────────────────
# API Call with Retry
# ──────────────────────────────────────────────────────────────────────────────

def api_call_with_retry(fn, max_retries=5, base_wait=5.0):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = base_wait * (2 ** attempt)
            print(f"      [RETRY {attempt+1}/{max_retries}] Error: {e}")
            print(f"      Waiting {wait:.0f}s before retry...")
            time.sleep(wait)


# ──────────────────────────────────────────────────────────────────────────────
# Output folders
# ──────────────────────────────────────────────────────────────────────────────

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def make_output_dirs(output_root):
    dirs = {
        "debug": ensure_dir(os.path.join(output_root, "debug")),
        "annotations": ensure_dir(os.path.join(output_root, "annotations")),
        "masks": ensure_dir(os.path.join(output_root, "masks")),
        "slice_masks": ensure_dir(os.path.join(output_root, "masks", "slice")),
        "root_masks": ensure_dir(os.path.join(output_root, "masks", "root")),
        "palette_masks": ensure_dir(os.path.join(output_root, "masks", "palette")),
    }
    return dirs


# ──────────────────────────────────────────────────────────────────────────────
# Debug Overlay
# ──────────────────────────────────────────────────────────────────────────────

def save_debug_overlay(img, palette_mask, slice_mask, root_mask, out_path):
    vis = img.copy()
    alpha = 0.40

    def blend(mask, color):
        nonlocal vis
        if np.count_nonzero(mask) == 0:
            return
        layer = np.zeros_like(vis)
        layer[mask > 0] = color
        vis = cv2.addWeighted(vis, 1.0, layer, alpha, 0)

    blend(palette_mask, (255, 140, 0))  # orange
    blend(slice_mask,   (0,   220, 0))  # green
    blend(root_mask,    (0,   0, 255))  # red

    h = vis.shape[0]
    legend = [
        ("Palette", (255, 140, 0)),
        ("Slice",   (0,   220, 0)),
        ("Root",    (0,   0, 255)),
    ]
    for i, (label, color) in enumerate(legend):
        y = (h - 25) - i * 30
        cv2.rectangle(vis, (12, y - 15), (34, y + 5), color, -1)
        cv2.putText(vis, label, (44, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(str(out_path), vis)


# ──────────────────────────────────────────────────────────────────────────────
# Per-image pipeline
# ──────────────────────────────────────────────────────────────────────────────

def process_image(img_path, model, dirs, wait_time=2.0):
    print(f"\n[{img_path.name}] Starting...")

    image  = Image.open(img_path).convert("RGB")
    cv_img = cv2.imread(str(img_path))
    image_size = image.size
    stem = img_path.stem

    prompts = {
        "slice":   "yam slice cross-section cut surface",
        "root":    "one single brown yam root",
        "palette": "color chart palette",
    }
    results = {}

    for i, (label, prompt) in enumerate(prompts.items()):
        print(f"  -> [{label}] prompt: '{prompt}'")

        point_result = api_call_with_retry(
            lambda p=prompt: model.point(image, p)
        )
        points = point_result.get("points", [])
        print(f"     Found {len(points)} point(s)")

        if label == "root" and len(points) > 1:
            print(f"     [ROOT] Multiple points found → keeping only the first one")
            points = points[:1]

        segments = []
        for pt in points:
            result = api_call_with_retry(
                lambda p=prompt, pt=pt: model.segment(
                    image,
                    p,
                    spatial_refs=[[pt["x"], pt["y"]]]
                )
            )
            segments.append({
                "path": result["path"],
                "bbox": result["bbox"],
            })
            time.sleep(0.3)

        results[label] = segments

        if i < len(prompts) - 1:
            print(f"     Waiting {wait_time}s before next prompt...")
            time.sleep(wait_time)

    print("  -> Rasterizing SVGs to masks...")
    slice_mask   = union_segments(results["slice"],   image_size)
    root_mask    = union_segments(results["root"],    image_size)
    palette_mask = union_segments(results["palette"], image_size)

    cv2.imwrite(os.path.join(dirs["slice_masks"],   f"{stem}_slice_mask.png"),   slice_mask)
    cv2.imwrite(os.path.join(dirs["root_masks"],    f"{stem}_root_mask.png"),    root_mask)
    cv2.imwrite(os.path.join(dirs["palette_masks"], f"{stem}_palette_mask.png"), palette_mask)

    save_debug_overlay(
        cv_img,
        palette_mask,
        slice_mask,
        root_mask,
        os.path.join(dirs["debug"], f"{stem}_debug_overlay.png")
    )

    with open(os.path.join(dirs["annotations"], f"{stem}_raw_segments.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"  [SAVED] {stem} done")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default="cleaned",           help="Input folder")
    parser.add_argument("--output",  default="moondream_results", help="Output folder")
    parser.add_argument("--api-key", required=True,               help="Moondream API key")
    parser.add_argument("--wait",    type=float, default=2.0,     help="Seconds between prompts")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    dirs = make_output_dirs(args.output)

    model = md.vl(api_key=args.api_key)

    valid_exts = {".jpg", ".jpeg", ".png", ".tif"}
    images = sorted([p for p in Path(args.input).iterdir() if p.suffix.lower() in valid_exts])
    print(f"Found {len(images)} images to process.")
    print(f"Output root: {args.output}")
    print("Folders:")
    print(f"  - {dirs['debug']}")
    print(f"  - {dirs['annotations']}")
    print(f"  - {dirs['slice_masks']}")
    print(f"  - {dirs['root_masks']}")
    print(f"  - {dirs['palette_masks']}")

    for img_path in images:
        try:
            process_image(img_path, model, dirs, wait_time=args.wait)
        except Exception as e:
            print(f"  [ERROR] Skipping {img_path.name}: {e}")
        print("  Waiting 3s before next image...")
        time.sleep(3.0)