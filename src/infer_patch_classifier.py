# src/infer_patch_classifier.py
import argparse, json, csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

def load_label_map(model_path: Path):
    # try sidecar labels json first
    lbl = model_path.with_suffix(".labels.json")
    if lbl.exists():
        with open(lbl) as f:
            data = json.load(f)
        classes = data.get("classes", [])
        return classes
    # else, try inside checkpoint
    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "classes" in ckpt:
        return ckpt["classes"]
    return []

def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device)
    # build arch that matches train script (MobileNetV2 head replaced)
    from torchvision import models
    m = models.mobilenet_v2(weights=None)
    # infer num_classes from label map
    classes = load_label_map(Path(model_path))
    num_classes = len(classes) if classes else 3
    in_f = m.classifier[1].in_features
    m.classifier[1] = torch.nn.Linear(in_f, num_classes)
    m.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    m.eval().to(device)
    return m, classes

def make_tf(img_size=96):
    # must match train transforms' normalization
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def read_kpts_csv(csv_path):
    rows = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def draw_dot(img, x, y, color, r=6, thickness=2):
    cv2.circle(img, (int(x), int(y)), r, color, thickness, lineType=cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to original .tif/.png/.jpg")
    ap.add_argument("--kpts_csv", required=True, help="CSV from export_keypoints.py")
    ap.add_argument("--model", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--out_overlay", default="data/processed/overlay_classified.png")
    ap.add_argument("--img_size", type=int, default=96)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--min_prob", type=float, default=0.40, help="Min softmax prob to accept nodule")
    ap.add_argument("--nodule_idx", type=int, default=None, help="Override class index for 'nodule'")
    ap.add_argument("--save_csv", default=None, help="Optional: write accepted nodules to CSV")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print("Device:", device)

    # load image to draw overlay
    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    # load model + classes
    model, classes = load_model(args.model, device)
    print("Classes:", classes if classes else "(unknown)")
    if args.nodule_idx is None:
        # auto-detect index of "nodule"
        if classes and "nodule" in classes:
            nodule_idx = classes.index("nodule")
        else:
            # fallback to common ordering ['background','fauna','nodule'] -> 2
            nodule_idx = 2
    else:
        nodule_idx = int(args.nodule_idx)
    print("Using nodule class index:", nodule_idx)

    # read keypoints
    krows = read_kpts_csv(args.kpts_csv)
    if not krows:
        print("No keypoints in CSV; nothing to classify.")
        return

    tf = make_tf(args.img_size)
    crops, metas = [], []
    for row in krows:
        cpath = row.get("crop_path", "")
        if not cpath or not Path(cpath).exists():
            # If crop not saved for some reason, skip
            continue
        try:
            pil = Image.open(cpath).convert("RGB")
        except Exception:
            continue
        crops.append(tf(pil))
        metas.append((float(row["x"]), float(row["y"]), float(row.get("size", 0.0)), cpath))

    if not crops:
        print("No readable crops; aborting.")
        return

    X = torch.stack(crops)  # [N,3,H,W]
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(X), args.batch):
            xb = X[i:i+args.batch].to(device)
            logits = model(xb)
            p = F.softmax(logits, dim=1)
            conf, cls = p.max(dim=1)
            preds.extend(cls.detach().cpu().tolist())
            probs.extend(conf.detach().cpu().tolist())

    # draw overlay and collect accepted nodules
    overlay = bgr.copy()
    accepted = []
    nodule_count = 0
    for (x, y, s, path), cls, conf in zip(metas, preds, probs):
        if (cls == nodule_idx) and (conf >= args.min_prob):
            draw_dot(overlay, x, y, (0, 255, 0), r=6, thickness=2)  # green
            nodule_count += 1
            accepted.append({"x": x, "y": y, "size": s, "prob": conf, "crop_path": path})
        else:
            draw_dot(overlay, x, y, (0, 0, 255), r=4, thickness=1)  # red

    # write overlay
    outp = Path(args.out_overlay)
    outp.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outp), overlay)

    # optional CSV of accepted nodules only
    if args.save_csv:
        with open(args.save_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["x","y","size","prob","crop_path"])
            w.writeheader()
            for r in accepted:
                w.writerow(r)

    print(f"Candidates: {len(metas)}")
    print(f"Accepted nodules: {nodule_count}")
    print(f"Min prob threshold: {args.min_prob}")
    print(f"Overlay saved → {outp}")
    if args.save_csv:
        print(f"Accepted nodules CSV → {args.save_csv}")

if __name__ == "__main__":
    main()
