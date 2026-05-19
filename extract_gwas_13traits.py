#!/usr/bin/env python3
"""
extract_gwas_13traits.py
========================
Extracts 13 GWAS traits from images using Moondream JSON segmentations.

Traits:
  sl_b_mean, sl_G_mean, sl_chroma_mean, sl_a_sd, sl_b_sd,
  sl_a_mean, sl_H_mean, sl_hue_angle_mean
  tu_fractal_dim, tu_major_axis_px, tu_roundness,
  tu_area_px, tu_solidity
"""

import cv2
import numpy as np
import json
import argparse
import csv
import math
import warnings
import re
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore")

SELECTED_TRAITS = [
    "sl_b_mean",
    "tu_fractal_dim",
    "sl_G_mean",
    "sl_chroma_mean",
    "tu_major_axis_px",
    "sl_a_sd",
    "sl_b_sd",
    "tu_roundness",
    "tu_area_px",
    "sl_a_mean",
    "tu_solidity",
    "sl_H_mean",
    "sl_hue_angle_mean",
]

# ─── MASK GENERATION ─────────────────────────────────────────────────────────

def _parse_svg_tokens(path_str: str) -> List:
    tokens = re.findall(r'[MmCcLlZz]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', path_str)
    commands, current_cmd, current_coords = [], None, []
    for tok in tokens:
        if tok in 'MmCcLlZz':
            if current_cmd is not None:
                commands.append((current_cmd, current_coords))
            current_cmd, current_coords = tok, []
        else:
            current_coords.append(float(tok))
    if current_cmd is not None:
        commands.append((current_cmd, current_coords))
    return commands

def _bezier(p0, p1, p2, p3, n=40):
    pts = []
    for t in np.linspace(0, 1, n):
        mt = 1 - t
        x = mt**3*p0[0] + 3*mt**2*t*p1[0] + 3*mt*t**2*p2[0] + t**3*p3[0]
        y = mt**3*p0[1] + 3*mt**2*t*p1[1] + 3*mt*t**2*p2[1] + t**3*p3[1]
        pts.append((x, y))
    return pts

def svg_path_to_mask(path_str: str, bbox: dict, img_w: int, img_h: int) -> np.ndarray:
    commands = _parse_svg_tokens(path_str)
    poly, cx, cy, sx, sy = [], 0.0, 0.0, 0.0, 0.0

    for cmd, coords in commands:
        if cmd == 'M': cx, cy = coords[0], coords[1]; sx, sy = cx, cy; poly.append((cx, cy))
        elif cmd == 'm': cx += coords[0]; cy += coords[1]; sx, sy = cx, cy; poly.append((cx, cy))
        elif cmd in ('C', 'c'):
            for s in range(len(coords) // 6):
                b = s * 6
                if cmd == 'C': p1, p2, p3 = (coords[b], coords[b+1]), (coords[b+2], coords[b+3]), (coords[b+4], coords[b+5])
                else: p1, p2, p3 = (cx+coords[b], cy+coords[b+1]), (cx+coords[b+2], cy+coords[b+3]), (cx+coords[b+4], cy+coords[b+5])
                poly.extend(_bezier((cx, cy), p1, p2, p3)[1:])
                cx, cy = p3
        elif cmd in ('L', 'l'):
            for s in range(len(coords) // 2):
                if cmd == 'L': nx, ny = coords[s*2], coords[s*2+1]
                else: nx, ny = cx+coords[s*2], cy+coords[s*2+1]
                poly.append((nx, ny)); cx, cy = nx, ny
        elif cmd in ('Z', 'z'): poly.append((sx, sy)); cx, cy = sx, sy

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not poly:
        return mask

    bx, by = bbox["x_min"], bbox["y_min"]
    bw, bh = bbox["x_max"] - bbox["x_min"], bbox["y_max"] - bbox["y_min"]
    pts = np.array(
        [(int(round((bx + x * bw) * img_w)), int(round((by + y * bh) * img_h))) for x, y in poly],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [pts], 255)
    return mask

def get_region_mask(annotation: dict, region_key: str, img_w: int, img_h: int) -> np.ndarray:
    combined = np.zeros((img_h, img_w), dtype=np.uint8)
    objects = annotation.get(region_key, [])
    if not isinstance(objects, list):
        return combined
    for obj in objects:
        p, bbox = obj.get("path", ""), obj.get("bbox", None)
        if p and bbox:
            combined = cv2.bitwise_or(combined, svg_path_to_mask(p, bbox, img_w, img_h))
    return combined

# ─── METRICS ─────────────────────────────────────────────────────────────────

def rgb_to_lab(pixels: np.ndarray) -> np.ndarray:
    img = pixels.reshape(-1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    lab[:, 0] = lab[:, 0] * 100.0 / 255.0
    lab[:, 1] -= 128.0
    lab[:, 2] -= 128.0
    return lab

def rgb_to_hsv(pixels: np.ndarray) -> np.ndarray:
    img = pixels.reshape(-1, 1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    hsv[:, 0] *= 2.0
    hsv[:, 1] /= 255.0
    hsv[:, 2] /= 255.0
    return hsv

def get_color_features(orig_rgb: np.ndarray, mask: np.ndarray, prefix: str) -> Dict[str, float]:
    if mask is None or not np.any(mask > 0):
        return {}
    pix = orig_rgb[mask > 0].astype(np.float32)
    if len(pix) == 0:
        return {}

    G = pix[:, 1]
    lab = rgb_to_lab(pix)
    hsv = rgb_to_hsv(pix)
    a, b = lab[:, 1], lab[:, 2]
    H = hsv[:, 0]

    ch = np.sqrt(a**2 + b**2)
    ha = np.degrees(np.arctan2(b, a))
    ha = np.where(ha < 0, ha + 360, ha)

    return {
        f"{prefix}_b_mean":         float(np.mean(b)),
        f"{prefix}_G_mean":         float(np.mean(G)),
        f"{prefix}_chroma_mean":    float(np.mean(ch)),
        f"{prefix}_a_sd":           float(np.std(a)),
        f"{prefix}_b_sd":           float(np.std(b)),
        f"{prefix}_a_mean":         float(np.mean(a)),
        f"{prefix}_H_mean":         float(np.mean(H)),
        f"{prefix}_hue_angle_mean": float(np.mean(ha)),
    }

def get_shape_features(mask: np.ndarray) -> Dict[str, float]:
    if mask is None or not np.any(mask > 0):
        return {}
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 50]
    if not cnts:
        return {}

    cnts.sort(key=cv2.contourArea, reverse=True)
    c = cnts[0]
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    hull = cv2.contourArea(cv2.convexHull(c)) + 1e-6

    if len(c) >= 5:
        _, (ma, mi), _ = cv2.fitEllipse(c)
        major = max(ma, mi)
    else:
        _, _, w, h = cv2.boundingRect(c)
        major = max(w, h)

    scales, coords = np.array([2, 4, 8, 16, 32, 64]), np.argwhere(mask > 0)
    fractal_dim = np.nan
    if len(coords) > 10:
        counts = [len(np.unique(coords // s, axis=0)) for s in scales]
        valid = np.array(counts) > 0
        if valid.sum() >= 3:
            fractal_dim = -float(np.polyfit(np.log(scales[valid]), np.log(np.array(counts)[valid]), 1)[0])

    return {
        "tu_fractal_dim":   fractal_dim,
        "tu_major_axis_px": float(major),
        "tu_roundness":     float((4 * math.pi * area) / (peri ** 2 + 1e-6)),
        "tu_area_px":       float(area),
        "tu_solidity":      float(area / hull),
    }

# ─── PIPELINE ────────────────────────────────────────────────────────────────

def extract_all(json_stem, img_dir, json_dir):
    json_path = json_dir / f"{json_stem}.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        annotation = json.load(f)

    acc_id = re.sub(r'(_clean|_raw|_segments)+$', '', json_stem)

    orig_rgb = None
    for suf in ["", "_clean", "_raw", "_clean_raw"]:
        for ext in [".jpg", ".jpeg", ".png", ".tif", ".JPG", ".PNG"]:
            p = img_dir / f"{acc_id}{suf}{ext}"
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None:
                    orig_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    break
        if orig_rgb is not None:
            break

    if orig_rgb is None:
        print(f"  [WARN] Image not found for {acc_id}")
        return None

    img_h, img_w = orig_rgb.shape[:2]

    sl_mask = get_region_mask(annotation, "slice", img_w, img_h)
    tu_mask = get_region_mask(annotation, "tub", img_w, img_h)
    if not np.any(tu_mask > 0):
        tu_mask = get_region_mask(annotation, "root", img_w, img_h)

    traits = {"stem": acc_id}
    for k in SELECTED_TRAITS:
        traits[k] = np.nan

    traits.update(get_color_features(orig_rgb, sl_mask, "sl"))
    traits.update(get_shape_features(tu_mask))

    return traits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Directory containing images")
    ap.add_argument("--json",  required=True, help="Directory containing JSON annotation files")
    ap.add_argument("--output", default="gwas_traits_13.csv", help="Output CSV path")
    args = ap.parse_args()

    img_dir  = Path(args.input)
    json_dir = Path(args.json)
    records  = []

    files = list(json_dir.glob("*.json"))
    print(f"Processing {len(files)} files for 13 traits...")

    for f in files:
        res = extract_all(f.stem, img_dir, json_dir)
        if res:
            records.append(res)
            print(
                f"✓ {res['stem']:<15} "
                f"a*: {res.get('sl_a_mean', float('nan')):.2f} | "
                f"b*: {res.get('sl_b_mean', float('nan')):.2f} | "
                f"roundness: {res.get('tu_roundness', float('nan')):.3f}"
            )

    if records:
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["stem"] + SELECTED_TRAITS, extrasaction="ignore")
            w.writeheader()
            for r in records:
                w.writerow({
                    k: ("" if isinstance(v, float) and math.isnan(v) else v)
                    for k, v in r.items()
                })
        print(f"\n[DONE] Saved {len(records)} entries to {args.output}")
    else:
        print("\n[WARN] No records extracted.")


if __name__ == "__main__":
    main()