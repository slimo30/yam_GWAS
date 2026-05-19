# GWAS 13 Traits — Documentation

This document describes all 13 phenotypic traits extracted from cassava/tuberous-root images. Traits are divided into two groups: **slice colour** (`sl_` prefix) and **tuberous root shape** (`tu_` prefix).

---

## Table of Contents

**Colour Traits (Slice)**

1. [sl_b_mean](#sl_b_mean)
2. [sl_a_mean](#sl_a_mean)
3. [sl_a_sd](#sl_a_sd)
4. [sl_b_sd](#sl_b_sd)
5. [sl_G_mean](#sl_g_mean)
6. [sl_chroma_mean](#sl_chroma_mean)
7. [sl_H_mean](#sl_h_mean)
8. [sl_hue_angle_mean](#sl_hue_angle_mean)

**Shape Traits (Tuberous Root)** 9. [tu_area_px](#tu_area_px) 10. [tu_major_axis_px](#tu_major_axis_px) 11. [tu_roundness](#tu_roundness) 12. [tu_solidity](#tu_solidity) 13. [tu_fractal_dim](#tu_fractal_dim)

---

## Colour Traits (Slice)

> All colour traits are computed from pixels inside the `slice` annotation mask.  
> Colour space conversions use OpenCV with L\* ∈ [0,100] and a\*, b\* ∈ [−128, 127].

---

### sl_b_mean

**Description**  
Mean b\* value of pixels within the slice region in the CIE L\*a\*b\* colour space.

**Biological Meaning**  
The b\* axis runs from blue (negative) to yellow (positive). Higher values indicate more yellow pigmentation in the flesh, often linked to carotenoid content.

**Computation**

1. Extract pixels inside the `slice` mask.
2. Convert RGB → CIE L\*a\*b\*.
3. Compute the arithmetic mean of the b\* channel.

**Units** — CIE b\* units (dimensionless, ≈ −128 to +127)  
**Expected Range** — −10 to +40 (white/cream flesh near 0; deep-yellow flesh toward +30–40)

```python
b = lab[:, 2]
"sl_b_mean": float(np.mean(b))
```

---

### sl_a_mean

**Description**  
Mean a\* value of pixels within the slice region in the CIE L\*a\*b\* colour space.

**Biological Meaning**  
The a\* axis runs from green (negative) to red (positive). Positive values indicate reddish or pinkish flesh; negative values indicate greenish tones.

**Computation**

1. Extract pixels inside the `slice` mask.
2. Convert RGB → CIE L\*a\*b\*.
3. Compute the arithmetic mean of the a\* channel.

**Units** — CIE a\* units (dimensionless, ≈ −128 to +127)  
**Expected Range** — −20 to +20

```python
a = lab[:, 1]
"sl_a_mean": float(np.mean(a))
```

---

### sl_a_sd

**Description**  
Standard deviation of the a\* channel across pixels within the slice region.

**Biological Meaning**  
Measures the uniformity of red–green pigmentation. Low values indicate homogeneous colour; high values suggest mottling, streaking, or mixed pigmentation.

**Computation**

1. Extract pixels inside the `slice` mask.
2. Convert RGB → CIE L\*a\*b\*.
3. Compute the population standard deviation of the a\* channel.

**Units** — CIE a\* units (dimensionless)  
**Expected Range** — 1–15; values above 20 suggest heterogeneous flesh colour

```python
a = lab[:, 1]
"sl_a_sd": float(np.std(a))
```

---

### sl_b_sd

**Description**  
Standard deviation of the b\* channel across pixels within the slice region.

**Biological Meaning**  
Measures yellow–blue pigmentation uniformity. High values indicate variable carotenoid distribution or colour gradients across the slice.

**Computation**

1. Extract pixels inside the `slice` mask.
2. Convert RGB → CIE L\*a\*b\*.
3. Compute the population standard deviation of the b\* channel.

**Units** — CIE b\* units (dimensionless)  
**Expected Range** — 2–20

```python
b = lab[:, 2]
"sl_b_sd": float(np.std(b))
```

---

### sl_G_mean

**Description**  
Mean value of the green channel (G) in the RGB colour space across pixels within the slice region.

**Biological Meaning**  
The raw green channel correlates with overall brightness and flesh greenness. In combination with a\* and b\*, it helps distinguish white/cream flesh from yellow or pigmented flesh.

**Computation**

1. Extract pixels inside the `slice` mask.
2. Read the G channel (index 1) from the RGB image (values 0–255).
3. Compute the arithmetic mean.

**Units** — Raw 8-bit intensity (0–255)  
**Expected Range** — 80–200; darker flesh tends toward lower values

```python
G = pix[:, 1]
"sl_G_mean": float(np.mean(G))
```

---

### sl_chroma_mean

**Description**  
Mean chroma (C\*) of the slice region, computed from CIE L\*a\*b\*.

**Biological Meaning**  
Chroma represents colour saturation — the distance from the neutral grey axis in Lab space. High values mean vivid, saturated flesh (e.g., deep yellow or orange); low values indicate pale or white flesh.

**Computation**

1. Extract pixels inside the `slice` mask.
2. Convert RGB → CIE L\*a\*b\*.
3. Per pixel: C\* = √(a\*² + b\*²).
4. Compute the arithmetic mean.

**Units** — CIE chroma units (dimensionless, ≥ 0)  
**Expected Range** — 5–50 (white/cream: 5–15; deep yellow: 25–50)

```python
ch = np.sqrt(a**2 + b**2)
"sl_chroma_mean": float(np.mean(ch))
```

---

### sl_H_mean

**Description**  
Mean hue (H) in the HSV colour space across pixels within the slice region.

**Biological Meaning**  
HSV hue angle describes the dominant colour tone. Distinguishes white/cream (near 0°), yellow (50–60°), orange (20–40°), and purple/pink (270–330°) flesh types.

**Computation**

1. Extract pixels inside the `slice` mask.
2. Convert RGB → HSV; rescale OpenCV hue from [0°, 180°] to [0°, 360°] by multiplying by 2.
3. Compute the arithmetic mean of H.

**Units** — Degrees (0°–360°)  
**Expected Range** — 30°–80° for typical yellow/cream flesh

```python
hsv[:, 0] *= 2.0        # OpenCV hue [0,180] → [0,360]
H = hsv[:, 0]
"sl_H_mean": float(np.mean(H))
```

---

### sl_hue_angle_mean

**Description**  
Mean hue angle (h°) computed from CIE L\*a\*b\* coordinates across pixels within the slice region.

**Biological Meaning**  
The Lab hue angle h° = arctan(b\*/a\*) is a perceptually uniform measure of colour tone, more precise than HSV hue for distinguishing subtle flesh colour differences relevant to nutritional quality (e.g., provitamin A carotenoids).

**Computation**

1. Extract pixels inside the `slice` mask.
2. Convert RGB → CIE L\*a\*b\*.
3. Per pixel: h° = degrees(arctan2(b\*, a\*)); wrap negative values by adding 360°.
4. Compute the arithmetic mean.

**Units** — Degrees (0°–360°): 0°/360° = red, 90° = yellow, 180° = green, 270° = blue  
**Expected Range** — Yellow/cream flesh: 80°–100°; orange flesh: 60°–80°

```python
ha = np.degrees(np.arctan2(b, a))
ha = np.where(ha < 0, ha + 360, ha)
"sl_hue_angle_mean": float(np.mean(ha))
```

---

## Shape Traits (Tuberous Root)

> All shape traits are computed from the largest contour found in the `tub` annotation mask (falls back to `root` if `tub` is empty).  
> Contours smaller than 50 px² are discarded as noise.

---

### tu_area_px

**Description**  
Contour area of the largest detected object in the tuberous root mask, in pixels.

**Biological Meaning**  
Proxy for root size. Larger pixel area indicates a larger or wider root. Useful for comparing relative root sizes across accessions when images are taken under standardised conditions.

**Computation**

1. Generate binary mask for the `tub`/`root` region.
2. Find external contours; discard those ≤ 50 px².
3. Select the largest contour; compute `cv2.contourArea(contour)`.

**Units** — Square pixels (px²); convert to cm² using image scale if known  
**Expected Range** — 5,000–200,000 px² for typical standardised setups

```python
area = cv2.contourArea(c)
"tu_area_px": float(area)
```

---

### tu_major_axis_px

**Description**  
Length of the major axis of the fitted ellipse of the largest contour in the tuberous root mask, in pixels.

**Biological Meaning**  
Approximates the maximum root length (longest dimension). Important for root shape characterisation — long, slender roots vs. short, round ones. Correlates with commercial grade and harvest index.

**Computation**

1. Select the largest contour from the `tub`/`root` mask.
2. If ≥ 5 contour points: fit an ellipse with `cv2.fitEllipse`; take the longer of the two axes.
3. If < 5 points: use `max(bounding_rect_width, bounding_rect_height)` as fallback.

**Units** — Pixels (px); convert to mm/cm using known image scale  
**Expected Range** — 100–1,500 px depending on imaging setup

```python
_, (ma, mi), _ = cv2.fitEllipse(c)
major = max(ma, mi)
"tu_major_axis_px": float(major)
```

---

### tu_roundness

**Description**  
Roundness (circularity) of the largest contour in the tuberous root mask.

**Biological Meaning**  
Describes how close the root outline is to a perfect circle. Round, compact roots score near 1.0; elongated or irregularly shaped roots score lower. Relevant for processing and peeling efficiency.

**Computation**  
Formula: **Roundness = (4π × Area) / Perimeter²**

1. Compute `area = cv2.contourArea(c)` and `peri = cv2.arcLength(c, closed=True)`.
2. Apply formula with +1e-6 epsilon to avoid division by zero.

**Units** — Dimensionless (0–1]: 1.0 = perfect circle; < 0.5 = elongated or irregular  
**Expected Range** — 0.3–0.85 for root crops

```python
"tu_roundness": float((4 * math.pi * area) / (peri ** 2 + 1e-6))
```

---

### tu_solidity

**Description**  
Solidity of the largest contour in the tuberous root mask — the ratio of contour area to convex hull area.

**Biological Meaning**  
Measures how convex the root shape is. Roots with deep constrictions, lobes, or surface irregularities score lower. High solidity (near 1.0) indicates a smooth, compact root desirable for processing.

**Computation**

1. Compute contour area and convex hull area (`cv2.convexHull`).
2. **Solidity = contour_area / hull_area** (+1e-6 epsilon on hull).

**Units** — Dimensionless (0–1]: 1.0 = perfectly convex; < 0.8 = notable irregularities  
**Expected Range** — 0.80–0.98 for root crops

```python
hull = cv2.contourArea(cv2.convexHull(c)) + 1e-6
"tu_solidity": float(area / hull)
```

---

### tu_fractal_dim

**Description**  
Fractal dimension of the tuberous root mask, estimated using the box-counting method.

**Biological Meaning**  
Quantifies the morphological complexity of the root outline. Higher values indicate more complex, irregular, or rough boundaries (branched or convoluted roots); lower values indicate smooth, simple shapes.

**Computation**  
Box-counting over the binary mask:

1. Define scales `[2, 4, 8, 16, 32, 64]` pixels.
2. For each scale `s`: count unique grid cells occupied (`np.unique(coords // s, axis=0)`).
3. Fit a line to log(scale) vs log(count); fractal dimension = **−slope**.
4. Requires ≥ 10 occupied pixels and ≥ 3 valid scale points; returns `NaN` otherwise.

**Units** — Dimensionless (1.0–2.0): ~1.0 = smooth curve; ~2.0 = highly complex boundary  
**Expected Range** — 1.1–1.6 for root crops

> **Note:** The estimate depends on image pixel resolution; compare only within images of the same resolution.

```python
scales = np.array([2, 4, 8, 16, 32, 64])
coords = np.argwhere(mask > 0)
counts = [len(np.unique(coords // s, axis=0)) for s in scales]
valid = np.array(counts) > 0
fractal_dim = -float(np.polyfit(np.log(scales[valid]), np.log(np.array(counts)[valid]), 1)[0])
"tu_fractal_dim": fractal_dim
```

---

_Generated from `extract_gwas_13traits.py`_
