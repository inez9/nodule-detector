# src/main.py
import argparse
from pathlib import Path
import cv2
from ops import (
    read_any_image_both, enhance_contrast, make_mask,
    color_outlier_mask, count_connected_components,
    detect_blobs, draw_keypoints
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to input image (.tif ok)")
    ap.add_argument("--outdir", default="data/processed", help="Where to save outputs")
    ap.add_argument("--invert", action="store_true", help="Use if nodules are darker than background")
    ap.add_argument("--purple_only", action="store_true", help="Filter only purple-ish objects")
    ap.add_argument("--fish_min_area", type=int, default=300, help="Min area to count a fish")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem

    # 1) Read both color + gray
    bgr, gray = read_any_image_both(args.image)
    cv2.imwrite(str(outdir / f"{stem}_gray.png"), gray)

    # 2) Build a mask of colored objects (fish/debris) and count them
    fish_mask = color_outlier_mask(bgr, purple_only=args.purple_only)
    fish_count, fish_kept = count_connected_components(fish_mask, min_area=args.fish_min_area)
    fish_overlay = bgr.copy()
    cnts, _ = cv2.findContours(fish_kept, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(fish_overlay, cnts, -1, (0,0,255), 2)
    cv2.putText(fish_overlay, f"Fish-like objects: {fish_count}", (12,28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)

    cv2.imwrite(str(outdir / f"{stem}_fishmask.png"), fish_kept)
    cv2.imwrite(str(outdir / f"{stem}_fishoverlay.png"), fish_overlay)

    # 3) Nodule pipeline
    enhanced = enhance_contrast(gray)
    cv2.imwrite(str(outdir / f"{stem}_enhanced.png"), enhanced)

    nodule_mask_raw = make_mask(enhanced, invert=args.invert)
    # Remove fish/debris from the nodule mask:
    fish_inv = cv2.bitwise_not(fish_kept)
    nodule_mask = cv2.bitwise_and(nodule_mask_raw, fish_inv)
    cv2.imwrite(str(outdir / f"{stem}_mask_raw.png"), nodule_mask_raw)
    cv2.imwrite(str(outdir / f"{stem}_mask_clean.png"), nodule_mask)

    # 4) Detect nodules on the cleaned mask
    bright_blobs = not args.invert
    keypoints = detect_blobs(nodule_mask, bright_blobs=bright_blobs,
                             min_area=30, max_area=5000, min_circ=0.4)
    nodule_overlay = draw_keypoints(bgr, keypoints, color=(0,255,0))
    cv2.imwrite(str(outdir / f"{stem}_overlay_nodules.png"), nodule_overlay)

    # 5) Report
    print(f"Fish-like objects: {fish_count}")
    print(f"Nodules (after color exclusion): {len(keypoints)}")
    print(f"Saved artifacts to {outdir.absolute()}")

if __name__ == "__main__":
    main()
