# src/label_patches.py
import argparse, cv2
from pathlib import Path
import shutil

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/patches/candidates")
    ap.add_argument("--dst", default="data/patches")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    for c in ["nodule","fauna","background"]:
        (dst/c).mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in src.iterdir() if p.suffix.lower() in [".png",".jpg",".jpeg",".tif",".tiff"]])
    i = 0
    while 0 <= i < len(files):
        p = files[i]
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            i += 1; continue
        show = img.copy()
        cv2.putText(show, f"{i+1}/{len(files)}  [1]=nodule  [2]=fauna  [3]=background  [a]=prev  [d]=next  [q]=quit",
                    (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("label", show)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('1'):
            shutil.move(str(p), str(dst/"nodule"/p.name)); i += 1
        elif k == ord('2'):
            shutil.move(str(p), str(dst/"fauna"/p.name)); i += 1
        elif k == ord('3'):
            shutil.move(str(p), str(dst/"background"/p.name)); i += 1
        elif k in (ord('a'), 81):  # left
            i = max(0, i-1)
        elif k in (ord('d'), 83):  # right
            i = min(len(files)-1, i+1)
        elif k == ord('q'):
            break
        else:
            i += 1
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

