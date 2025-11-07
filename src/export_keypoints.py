# src/export_keypoints.py
import argparse, csv
from pathlib import Path
import cv2
from ops import read_any_image_both, enhance_contrast, make_mask, color_outlier_mask, detect_blobs


import math

def nms_points(kps, min_dist=20):
    """Keep at most one blob within min_dist px (largest kept)."""
    kept = []
    pts = sorted(kps, key=lambda k: k.size, reverse=True)
    r2 = min_dist * min_dist
    for kp in pts:
        x, y = kp.pt
        if all((x - p.pt[0])**2 + (y - p.pt[1])**2 >= r2 for p in kept):
            kept.append(kp)
    return kept


def safe_crop(img, cx, cy, size):
    half = size // 2
    y0, y1 = max(0, cy - half), min(img.shape[0], cy + half)
    x0, x1 = max(0, cx - half), min(img.shape[1], cx + half)
    crop = img[y0:y1, x0:x1]
    # pad if near borders
    if crop.shape[0] != size or crop.shape[1] != size:
        crop = cv2.copyMakeBorder(
            crop,
            top=max(0, size - crop.shape[0] - 0)//2,
            bottom=max(0, size - crop.shape[0])//2,
            left=max(0, size - crop.shape[1] - 0)//2,
            right=max(0, size - crop.shape[1])//2,
            borderType=cv2.BORDER_REFLECT_101
        )
    return crop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude_color", action="store_true")
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--min_dist", type=int, default=0, help="NMS distance in px; 0=off")


    # NEW: tuning knobs
    ap.add_argument("--min_area", type=int, default=150)   # raise to reduce detections
    ap.add_argument("--max_area", type=int, default=8000)
    ap.add_argument("--min_circ", type=float, default=0.55) # raise to prefer round blobs
    ap.add_argument("--patch", type=int, default=96)        # crop size around each blob
    args = ap.parse_args()

    outdir = Path(args.out).parent; outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.out).stem

    bgr, gray = read_any_image_both(args.image)
    enhanced = enhance_contrast(gray)

    # light denoise to kill tiny speckles
    den = cv2.GaussianBlur(enhanced, (3,3), 0)

    mask = make_mask(den, invert=args.invert)
    if args.exclude_color:
        color_mask = color_outlier_mask(bgr, purple_only=False)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(color_mask))

    # morphology to remove salt-and-pepper noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # blob detect with stricter thresholds
    kps = detect_blobs(
        mask,
        bright_blobs=not args.invert,
        min_area=args.min_area,
        max_area=args.max_area,
        min_circ=args.min_circ
    )

    if args.min_dist and args.min_dist > 0:
        before = len(kps)
        kps = nms_points(kps, min_dist=args.min_dist)
        print(f"NMS pruned {before - len(kps)} near-duplicates (min_dist={args.min_dist})")

    print(f"Detected {len(kps)} candidate blobs")

    # Full-image overlay so you can SEE them
    overlay = bgr.copy()
    for kp in kps:
        x, y, s = int(kp.pt[0]), int(kp.pt[1]), int(kp.size/2)
        cv2.circle(overlay, (x, y), max(2, s//2), (0,255,0), 2)
    cv2.imwrite(str(outdir / f"{stem}_overlay_kps.png"), overlay)

    # Save crops + CSV
    patch_dir = outdir / f"{stem}_patches"; patch_dir.mkdir(exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["x","y","size","crop_path"])
        for i, kp in enumerate(kps):
            cx, cy, size = int(kp.pt[0]), int(kp.pt[1]), int(max(args.patch, kp.size))
            crop = safe_crop(bgr, cx, cy, args.patch)
            cpath = patch_dir / f"kp_{i:05d}.png"
            cv2.imwrite(str(cpath), crop)
            w.writerow([kp.pt[0], kp.pt[1], kp.size, str(cpath)])

    print(f"Saved keypoints CSV → {args.out}")
    print(f"Saved crops     → {patch_dir}")
    print(f"Saved overlay   → {outdir / (stem + '_overlay_kps.png')}")

if __name__ == "__main__":
    main()
