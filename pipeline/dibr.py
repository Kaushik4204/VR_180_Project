import cv2
import numpy as np

def _curve_x_map(width, curvature):
    """
    Build an x-mapping that bends the image horizontally to simulate a curved screen.
    curvature in [0..0.6]. 0 = flat.
    """
    xs = np.linspace(-1, 1, width, dtype=np.float32)
    if curvature <= 1e-6:
        return xs
    # Simple quadratic bend; could be replaced with a better screen-space warp
    bend = xs + curvature * (xs**3 - xs)
    # Normalize back to [-1,1]
    bend = bend / np.max(np.abs(bend))
    return bend

def stereo_from_depth(
    image_path,
    depth,
    out_left_path,
    out_right_path,
    baseline_px=6.0,
    curvature=0.35,
    inpaint_radius=3
):
    """
    Create left/right views from a single image using a depth map.
    - depth: 2D numpy array in [0,1], higher = nearer
    - baseline_px: pixel disparity at mid-depth (depth=0.5)
    - curvature: screen bend factor (0=flat)
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    # Curved-screen warping (pre-warp)
    xs = _curve_x_map(w, curvature)
    grid_x = np.tile((xs + 1) * 0.5 * (w - 1), (h, 1)).astype(np.float32)
    grid_y = np.tile(np.linspace(0, h - 1, h, dtype=np.float32)[:, None], (1, w))
    img_curved = cv2.remap(img, grid_x, grid_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    # Disparity: proportional to (depth - 0.5)
    disp = baseline_px * (depth - 0.5)
    disp = cv2.GaussianBlur(disp.astype(np.float32), (5,5), 0)

    # Build flow fields for L/R
    # Left eye shifts content to the right for near objects (negative shift moves pixels left)
    flow_x_left  = -disp
    flow_x_right =  disp

    # Remap to get L/R images
    base_x = np.tile(np.arange(w, dtype=np.float32), (h,1))
    base_y = np.tile(np.arange(h, dtype=np.float32)[:,None], (1,w))

    map_left_x  = base_x + flow_x_left
    map_right_x = base_x + flow_x_right

    left  = cv2.remap(img_curved, map_left_x,  base_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    right = cv2.remap(img_curved, map_right_x, base_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Create masks of holes (zeros) to inpaint
    mask_l = (left.sum(axis=2) == 0).astype(np.uint8) * 255
    mask_r = (right.sum(axis=2) == 0).astype(np.uint8) * 255

    if inpaint_radius > 0:
        left  = cv2.inpaint(left,  mask_l, inpaint_radius, cv2.INPAINT_TELEA)
        right = cv2.inpaint(right, mask_r, inpaint_radius, cv2.INPAINT_TELEA)

    cv2.imwrite(str(out_left_path), left)
    cv2.imwrite(str(out_right_path), right)
