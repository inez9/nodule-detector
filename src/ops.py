import cv2
import numpy as np
from pathlib import Path

def read_any_image_both(path: str):
    """Return (bgr, gray8). Works for 16-bit TIFFs too."""
    bgr = cv2.imread(str(Path(path)), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise FileNotFoundError(f"Could not read {path}")

    # If grayscale single-channel, synthesize BGR for drawing overlays
    if bgr.ndim == 2:
        gray = bgr
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        bgr_for_overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return bgr_for_overlay, gray

    # If multi-channel, ensure we have an 8-bit gray for processing
    if bgr.dtype != np.uint8:
        tmp = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(tmp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        bgr8 = cv2.normalize(bgr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return bgr8, gray
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return bgr, gray

def color_outlier_mask(bgr: np.ndarray, purple_only: bool = True):
    """
    Returns a binary mask (255=colored object) for likely 'fish/debris' regions.
    Strategy: high saturation (S) + optional purple/magenta hue range.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Thresholds you can tweak:
    # Saturation: > 50 (on 0..255 scale) means colored. Raise/lower as needed.
    s_min = 50
    v_min = 30  # avoid counting very dark noise

    if purple_only:
        # OpenCV H is 0..180. Purples/magentas are roughly 120..170.
        lower = np.array([120, s_min, v_min], dtype=np.uint8)
        upper = np.array([170, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
    else:
        # Any strongly colored pixel (not gray/brown)
        h, s, v = cv2.split(hsv)
        mask = (s > s_min) & (v > v_min)
        mask = mask.astype(np.uint8) * 255

    # Clean up: remove specks, fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8), iterations=1)
    return mask

def count_connected_components(mask: np.ndarray, min_area: int = 300):
    """Count connected components above a size threshold."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # stats rows: [label, x, y, w, h, area] but OpenCV returns just stats (no label column)
    # stats shape: (num, 5) -> [x, y, w, h, area]
    count = 0
    keep = np.zeros_like(mask)
    for i in range(1, num):  # skip background 0
        area = stats[i, 4]
        if area >= min_area:
            keep[labels == i] = 255
            count += 1
    return count, keep

def enhance_contrast(gray: np.ndarray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def make_mask(enhanced: np.ndarray, invert=False):
    blurred = cv2.GaussianBlur(enhanced, (5,5), 0)
    mask = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY if not invert else cv2.THRESH_BINARY_INV,
        31, -5
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)
    return mask

def detect_blobs(mask: np.ndarray, bright_blobs=True, min_area=30, max_area=5000, min_circ=0.4):
    p = cv2.SimpleBlobDetector_Params()
    p.filterByArea = True;        p.minArea = float(min_area); p.maxArea = float(max_area)
    p.filterByCircularity = True; p.minCircularity = float(min_circ)
    p.filterByConvexity = False;  p.filterByInertia = False
    p.filterByColor = True;       p.blobColor = 255 if bright_blobs else 0
    return cv2.SimpleBlobDetector_create(p).detect(mask)

def draw_keypoints(bgr: np.ndarray, keypoints, color=(0,255,0)):
    out = bgr.copy()
    for k in keypoints:
        x, y, r = int(k.pt[0]), int(k.pt[1]), int(k.size/2)
        cv2.circle(out, (x,y), r, color, 2)
        cv2.circle(out, (x,y), 2, (0,0,255), -1)
    cv2.putText(out, f"Count: {len(keypoints)}", (12,28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return out
