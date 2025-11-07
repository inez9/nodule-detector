#crops & balances datasets
# src/make_patches.py
import cv2, os
from pathlib import Path
import numpy as np
from ops import read_any_image_both, enhance_contrast, make_mask, detect_blobs, color_outlier_mask

def save_crop(img, cx, cy, size, out_dir, stem, idx):
    h, w = img.shape[:2]
    r = size // 2
    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(w, cx + r), min(h, cy + r)
    crop = img[y1:y2, x1:x2]
    if crop.shape[0] < size or crop.shape[1] < size:
        pad = np.zeros((size, size, 3), dtype=img.dtype)
        pad[:crop.shape[0], :crop.shape[1]] = crop
        crop = pad
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{stem}_{idx:04d}.png"), crop)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="data/patches/candidates")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--invert", action="store_true")
    args = ap.parse_args()

    bgr, gray = read_any_image_both(args.image)
    enhanced = enhance_contrast(gray)
    mask = make_mask(enhanced, invert=args.invert)

    # optional: remove colored fauna
    fish = color_outlier_mask(bgr, purple_only=True)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(fish))

    kps = detect_blobs(mask, bright_blobs=not args.invert)
    out_dir = Path(args.out)

    for i, k in enumerate(kps):
        cx, cy = int(k.pt[0]), int(k.pt[1])
        save_crop(bgr, cx, cy, args.size, out_dir, Path(args.image).stem, i)

if __name__ == "__main__":
    main()
