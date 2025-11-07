# summarize_batch.py
# Merge per-image results from *_nodules_{suffix}.csv into one summary with coverage (%)

import argparse, csv, math
from pathlib import Path
import cv2

def iter_nodule_csvs(processed_dir: Path, suffix: str):
    # Match files like: <stem>_nodules_<suffix>.csv
    pattern = f"*_nodules_{suffix}.csv" if suffix else "*_nodules_*.csv"
    yield from sorted(processed_dir.glob(pattern))

def image_hw(raw_dir: Path, stem: str):
    # Try to find the original image (assumes .tif)
    img_path = raw_dir / f"{stem}.tif"
    if not img_path.exists():
        return None, None, None
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None, None
    h, w = img.shape[:2]
    return img_path, h, w

def summarize_file(csv_path: Path):
    rows = []
    with csv_path.open() as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                d = float(r.get("size", "0"))  # kp.size ~ diameter in px
                p = float(r.get("prob", "0"))
            except ValueError:
                d, p = 0.0, 0.0
            rows.append((d, p))
    if not rows:
        return 0, 0.0, 0.0, 0.0
    n = len(rows)
    total_px_area = sum(math.pi * (d/2.0)**2 for d, _ in rows)
    mean_diam_px  = sum(d for d, _ in rows) / n
    mean_prob     = sum(p for _, p in rows) / n
    return n, total_px_area, mean_diam_px, mean_prob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed", help="Where *_nodules_*.csv live")
    ap.add_argument("--raw_dir",       default="data/raw",       help="Where the .tif images live")
    ap.add_argument("--suffix",        default="p040",           help="Suffix used when saving nodules CSVs (e.g., p040)")
    ap.add_argument("--out",           default="batch_summary.csv", help="Output CSV filename (written into processed_dir)")
    args = ap.parse_args()

    processed = Path(args.processed_dir)
    raw = Path(args.raw_dir)
    out_csv = processed / args.out

    csv_files = list(iter_nodule_csvs(processed, args.suffix))
    if not csv_files:
        print(f"[warn] no *_nodules_{args.suffix}.csv found in {processed}")
        return

    print(f"[info] found {len(csv_files)} nodules CSVs with suffix '{args.suffix}'")

    records = []
    grand_n = 0
    grand_area = 0.0
    for f in csv_files:
        # derive stem: "<stem>_nodules_<suffix>"
        stem = f.stem.replace(f"_nodules_{args.suffix}", "")
        n, pixel_area, mean_diam, mean_prob = summarize_file(f)
        img_path, h, w = image_hw(raw, stem)

        if h and w:
            coverage_pct = 100.0 * pixel_area / (h * w)
            hw_str = f"{w}x{h}"
        else:
            coverage_pct = None
            hw_str = ""

        records.append({
            "image": stem,
            "image_px": hw_str,
            "accepted_nodules": n,
            "est_area_px": round(pixel_area, 1),
            "coverage_pct": round(coverage_pct, 4) if coverage_pct is not None else "",
            "mean_diam_px": round(mean_diam, 2),
            "mean_prob": round(mean_prob, 4),
            "source_csv": str(f.relative_to(processed))
        })
        grand_n += n
        grand_area += pixel_area

    # Write summary CSV
    fieldnames = ["image","image_px","accepted_nodules","est_area_px","coverage_pct",
                  "mean_diam_px","mean_prob","source_csv"]
    with out_csv.open("w", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(r)

    print(f"✓ Saved summary → {out_csv}")
    # Print quick totals (only meaningful if images are same size)
    print(f"Totals: accepted_nodules={grand_n}, est_area_px={grand_area:.0f}")

if __name__ == "__main__":
    main()
