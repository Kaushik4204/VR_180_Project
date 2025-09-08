import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def to_tb_canvas_2to1(left_dir: Path, right_dir: Path, out_dir: Path, target_height_2to1=1440):
    """
    Packs L/R images onto a 2:1 canvas (W=2H) as Top/Bottom (left on top, right on bottom).
    This matches the common VR180 container size; set your VR player to 180° + TB.
    """
    lefts = sorted(Path(left_dir).glob("*.png"))
    rights = sorted(Path(right_dir).glob("*.png"))
    assert len(lefts) == len(rights) and len(lefts) > 0, "Mismatched L/R frames"

    H2 = target_height_2to1
    W2 = H2 * 2

    for l, r in tqdm(zip(lefts, rights), total=len(lefts), desc="Packing VR180 TB canvas"):
        li = cv2.imread(str(l))
        ri = cv2.imread(str(r))
        # Fit each eye to half the canvas height (keep aspect, pad)
        h, w = li.shape[:2]
        scale = min(W2 / w, (H2 // 2) / h)
        new_w, new_h = int(w * scale), int(h * scale)
        li_res = cv2.resize(li, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ri_res = cv2.resize(ri, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((H2, W2, 3), dtype=np.uint8)
        # Top: left eye
        x_off = (W2 - new_w)//2
        y_off = (H2//2 - new_h)//2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = li_res

        # Bottom: right eye
        y2_off = H2//2 + (H2//2 - new_h)//2
        canvas[y2_off:y2_off+new_h, x_off:x_off+new_w] = ri_res

        out_path = Path(out_dir) / l.name
        cv2.imwrite(str(out_path), canvas)
