# batch.py
# Run export_keypoints.py + infer_patch_classifier.py over a folder of .tif files
# Usage example (copy-paste):
#   uv run python batch.py \
#     --raw_dir data/raw \
#     --processed_dir data/processed \
#     --model models/patch_clf_mobilenet.pt \
#     --min_area 70 --max_area 15000 --min_circ 0.55 --min_dist 12 --patch 128 \
#     --min_prob 0.40 --exclude_color

import argparse, subprocess, sys, csv
from pathlib import Path

def run(cmd: list[str]) -> int:
    print(">>", " ".join(cmd))
    return subprocess.run(cmd).returncode

def csv_has_rows(csv_path: Path) -> bool:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return False
    try:
        with csv_path.open() as f:
            rdr = csv.reader(f)
            rows = list(rdr)
            # header + at least one data row
            return len(rows) >= 2
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir",        default="data/raw", help="Folder with .tif files")
    ap.add_argument("--processed_dir",  default="data/processed", help="Output folder")
    ap.add_argument("--model",          required=True, help="Path to .pt model")

    # export_keypoints knobs
    ap.add_argument("--min_area", type=int, default=70)
    ap.add_argument("--max_area", type=int, default=15000)
    ap.add_argument("--min_circ", type=float, default=0.55)
    ap.add_argument("--patch",    type=int, default=128)
    ap.add_argument("--invert",   action="store_true", help="If nodules are darker")
    ap.add_argument("--exclude_color", action="store_true", help="HSV filter to drop colorful fauna")
    ap.add_argument("--min_dist", type=int, default=0, help="NMS distance in px; 0=off (requires you added it to export_keypoints.py)")

    # infer knobs
    ap.add_argument("--min_prob", type=float, default=0.40, help="Min softmax prob to accept nodule")
    ap.add_argument("--img_size", type=int,   default=96)
    ap.add_argument("--suffix",   default="p040", help="Overlay/CSV suffix tag (e.g., p040 for prob=0.40)")

    # control
    ap.add_argument("--limit", type=int, default=0, help="Process only first N images (0 = all)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tifs = sorted(raw_dir.glob("*.tif"))

    if args.limit > 0:
        tifs = tifs[:args.limit]
    if not tifs:
        print(f"[warn] no .tif files found in {raw_dir}")
        sys.exit(0)

    print(f"[info] found {len(tifs)} .tif files in {raw_dir}")
    for tif in tifs:
        stem = tif.stem
        csv_out   = out_dir / f"{stem}_kpts.csv"
        overlay   = out_dir / f"{stem}_overlay_{args.suffix}.png"
        nod_csv   = out_dir / f"{stem}_nodules_{args.suffix}.csv"

        print(f"\n=== Processing: {stem} ===")
        if nod_csv.exists():
            print(f"[skip] already processed {stem}")
            continue

        # 1) export keypoints
        export_cmd = [
            sys.executable, "src/export_keypoints.py",
            "--image", str(tif),
            "--out",   str(csv_out),
            "--min_area", str(args.min_area),
            "--max_area", str(args.max_area),
            "--min_circ", str(args.min_circ),
            "--patch",    str(args.patch),
        ]
        if args.exclude_color: export_cmd.append("--exclude_color")
        if args.invert:        export_cmd.append("--invert")
        if args.min_dist and args.min_dist > 0:
            # only works if you added --min_dist + NMS to export_keypoints.py
            export_cmd += ["--min_dist", str(args.min_dist)]

        if args.dry_run:
            print("[dry] would run export")
        else:
            rc = run(export_cmd)
            if rc != 0:
                print(f"[skip] export failed (rc={rc}) for {stem}")
                continue

        # 2) sanity: CSV must exist & have rows
        if not csv_has_rows(csv_out):
            print(f"[skip] no keypoints produced for {stem} → {csv_out}")
            continue

        # 3) run classifier
        infer_cmd = [
            sys.executable, "src/infer_patch_classifier.py",
            "--image",       str(tif),
            "--kpts_csv",    str(csv_out),
            "--model",       str(args.model),
            "--out_overlay", str(overlay),
            "--min_prob",    str(args.min_prob),
            "--img_size",    str(args.img_size),
            "--save_csv",    str(nod_csv),
        ]
        if args.dry_run:
            print("[dry] would run infer")
        else:
            rc = run(infer_cmd)
            if rc != 0:
                print(f"[warn] infer failed (rc={rc}) for {stem}")

    print("\n[done] batch complete.")

if __name__ == "__main__":
    main()

