# app.py
import streamlit as st
from pathlib import Path
import sys, subprocess, csv, math, cv2

st.set_page_config(page_title="Nodule Detector", layout="wide")
st.title("🌊 Automated Nodule Detector")

uploaded = st.file_uploader("Upload a seafloor image (.tif/.tiff/.png/.jpg)", type=["tif","tiff","png","jpg","jpeg"])
colL, colR = st.columns([1,1])

if uploaded:
    data_dir = Path("data"); (data_dir / "processed").mkdir(parents=True, exist_ok=True)
    tmp = data_dir / "temp_input.tif"
    with open(tmp, "wb") as f: f.write(uploaded.getbuffer())

    with colL:
        st.image(str(tmp), caption="Original", use_container_width=True)

    if st.button("Run analysis"):
        with st.spinner("Detecting candidates and classifying…"):
            stem = tmp.stem
            kcsv = data_dir/"processed"/f"{stem}_kpts.csv"
            overlay = data_dir/"processed"/f"{stem}_overlay_p040.png"
            nodcsv = data_dir/"processed"/f"{stem}_nodules_p040.csv"

            # 1) export keypoints
            subprocess.run([
                sys.executable, "src/export_keypoints.py",
                "--image", str(tmp),
                "--out", str(kcsv),
                "--min_area","70","--max_area","15000","--min_circ","0.55",
                "--patch","128","--exclude_color"
            ], check=False)

            # 2) classify
            subprocess.run([
                sys.executable, "src/infer_patch_classifier.py",
                "--image", str(tmp),
                "--kpts_csv", str(kcsv),
                "--model","models/patch_clf_mobilenet.pt",
                "--out_overlay", str(overlay),
                "--min_prob","0.40",
                "--save_csv", str(nodcsv)
            ], check=False)

            # 3) coverage stats
            total_px=0; n_count=0
            try:
                with open(nodcsv) as f:
                    for r in csv.DictReader(f):
                        d = float(r["size"]); total_px += math.pi*(d/2)**2; n_count += 1
                img = cv2.imread(str(tmp)); h,w = img.shape[:2]
                cov = 100*total_px/(h*w)
            except Exception:
                cov, n_count = 0.0, 0

        with colR:
            if Path(overlay).exists():
                st.image(str(overlay), caption=f"Overlay — nodules: {n_count}, coverage: {cov:.3f}%", use_container_width=True)
            else:
                st.warning("No overlay generated. Try another image.")

