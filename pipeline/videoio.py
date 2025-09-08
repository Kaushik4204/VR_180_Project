import subprocess
from pathlib import Path
import imageio
import cv2
from tqdm import tqdm

def extract_frames(video_path, out_dir: Path, target_height=720, fps=None):
    """
    Extract frames as PNGs using ffmpeg. Optionally downscale height and override FPS.
    Returns {'fps': float, 'count': int}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Probe FPS
    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    orig_fps = meta.get('fps', 24)
    reader.close()

    use_fps = orig_fps if (not fps or fps <= 0) else fps

    # ffmpeg scaling
    scale_str = f"scale=-2:{target_height}"
    fps_str = f",fps={use_fps}" if use_fps and use_fps > 0 else ""

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"{scale_str}{fps_str}",
        str(out_dir / "%06d.png")
    ]
    subprocess.run(cmd, check=True)

    count = len(list(out_dir.glob("*.png")))
    return {"fps": use_fps, "count": count}

def assemble_video(left_dir: Path, right_dir: Path, out_mp4: Path, layout="sbs", fps=24):
    """
    Combine L/R frames into a single video.
    layout: 'sbs' or 'tb'
    """
    lefts = sorted(Path(left_dir).glob("*.png"))
    rights = sorted(Path(right_dir).glob("*.png"))
    assert len(lefts) == len(rights) and len(lefts) > 0, "Mismatched L/R frames"

    # Read first to get size
    sample = cv2.imread(str(lefts[0]))
    h, w = sample.shape[:2]

    writer = imageio.get_writer(out_mp4, fps=fps, codec='libx264', quality=8)

    for l, r in tqdm(zip(lefts, rights), total=len(lefts), desc=f"Writing {out_mp4.name}"):
        li = cv2.imread(str(l))
        ri = cv2.imread(str(r))
        if layout == "sbs":
            frame = cv2.hconcat([li, ri])
        else:
            frame = cv2.vconcat([li, ri])
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    writer.close()

def make_preview_gif(video_path, out_gif):
    """
    Create a short animated GIF preview from the SBS video.
    """
    import imageio

    reader = imageio.get_reader(video_path)
    frames = []
    max_frames = 40
    for i, frame in enumerate(reader):
        if i >= max_frames:
            break
        frames.append(frame)
    reader.close()

    if frames:
        imageio.mimsave(out_gif, frames, duration=0.06)
