import asyncio
import sys

# ✅ Windows asyncio fix
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
import tempfile
import uuid
from pathlib import Path
import gradio as gr

# 🔹 Import pipeline modules
from pipeline.videoio import extract_frames, assemble_video, make_preview_gif
from pipeline.depth import DepthEstimator
from pipeline.dibr import stereo_from_depth
from pipeline.vr180_pack import to_tb_canvas_2to1

# 🔹 Outputs folder (persisted results)
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# 🔹 Load depth model once (lazy initialization happens on first inference)
depth_model = DepthEstimator()

def convert_video(
    video_file,
    target_height=720,
    fps=None,
    stereo_baseline=6.0,
    curvature=0.35,
    inpaint_radius=3,
    make_vr180=True,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Converts a 2D video into:
    - Side-by-Side (SBS) 3D
    - Top-Bottom (TB) 3D
    - Optional VR180 TB (2:1 format for headsets)

    Returns:
        preview_gif, sbs_path, tb_path, vr180_path (if enabled)
    """
    if video_file is None:
        raise gr.Error("Please upload a video file.")

    # Working directory for this job
    job_id = uuid.uuid4().hex[:8]
    workdir = Path(tempfile.mkdtemp(prefix=f"job_{job_id}_"))
    frames_dir = workdir / "frames"
    left_dir = workdir / "left"
    right_dir = workdir / "right"
    for d in (frames_dir, left_dir, right_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1️⃣ Extract frames
    info = extract_frames(video_file, frames_dir, target_height=target_height, fps=fps)

    # 2️⃣ Depth estimation + stereo synthesis
    progress(0, desc="Estimating depth & generating stereo frames...")
    for fpath in progress.tqdm(sorted(frames_dir.glob("*.png")), desc="Frames"):
        depth = depth_model.infer(fpath)
        stereo_from_depth(
            image_path=fpath,
            depth=depth,
            out_left_path=left_dir / fpath.name,
            out_right_path=right_dir / fpath.name,
            baseline_px=stereo_baseline,
            curvature=curvature,
            inpaint_radius=inpaint_radius,
        )

    # 3️⃣ Assemble SBS & TB videos
    base = Path(video_file).stem
    sbs_mp4 = OUT_DIR / f"{base}_{job_id}_sbs.mp4"
    tb_mp4 = OUT_DIR / f"{base}_{job_id}_tb.mp4"
    assemble_video(left_dir, right_dir, sbs_mp4, layout="sbs", fps=info["fps"])
    assemble_video(left_dir, right_dir, tb_mp4, layout="tb", fps=info["fps"])

    # 4️⃣ Optional: VR180 (Top-Bottom on 2:1 canvas)
    vr180_tb_mp4 = None
    if make_vr180:
        vr180_dir = workdir / "vr180_tb"
        vr180_dir.mkdir(exist_ok=True)
        to_tb_canvas_2to1(left_dir, right_dir, vr180_dir, target_height_2to1=1440)
        vr180_tb_mp4 = OUT_DIR / f"{base}_{job_id}_vr180_tb.mp4"
        assemble_video(vr180_dir, vr180_dir, vr180_tb_mp4, layout="tb", fps=info["fps"])

    # 5️⃣ Generate animated GIF preview
    preview_gif = OUT_DIR / f"{base}_{job_id}_preview.gif"
    make_preview_gif(sbs_mp4, preview_gif)

    return (
        str(preview_gif),
        str(sbs_mp4),
        str(tb_mp4),
        (str(vr180_tb_mp4) if vr180_tb_mp4 else None),
    )

# ================================
# 🚀 Gradio UI
# ================================
with gr.Blocks(title="VR180 Inception Converter — MVP") as demo:
    gr.Markdown(
        """
        # 🎬 VR180 Inception Converter — MVP
        Upload a 2D clip and get:
        - **3D SBS** (Side-by-Side)
        - **Top-Bottom 3D**
        - **VR180-friendly TB (2:1)** for VR headsets

        🔧 Adjust parameters for best results:
        - **Stereo Baseline** → Controls depth strength  
        - **Curvature** → Bends the virtual screen (0 = flat, 0.5 = very curved)  
        - Use `*_vr180_tb.mp4` in a VR player set to **180° / TB**
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            in_video = gr.Video(label="🎥 Upload 2D video", sources=["upload", "webcam"])
            target_h = gr.Slider(360, 1080, value=720, step=60, label="Target processing height (px)")
            fps_in = gr.Slider(0, 60, value=0, step=1, label="Override FPS (0 = keep original)")
            stereo = gr.Slider(0.0, 12.0, value=6.0, step=0.5, label="Stereo baseline (px)")
            curv = gr.Slider(0.0, 0.6, value=0.35, step=0.05, label="Curvature")
            inpaint = gr.Slider(1, 7, value=3, step=1, label="Inpaint radius (px)")
            make_vr180 = gr.Checkbox(value=True, label="Export VR180-friendly TB (2:1)")

            run_btn = gr.Button("🚀 Convert", variant="primary")
        with gr.Column(scale=1):
            preview = gr.Image(label="Preview GIF (SBS)", type="filepath")
            sbs_path = gr.File(label="⬇ Download: 3D SBS MP4")
            tb_path = gr.File(label="⬇ Download: 3D Top-Bottom MP4")
            vr180_path = gr.File(label="⬇ Download: VR180 TB MP4")

    run_btn.click(
        convert_video,
        inputs=[in_video, target_h, fps_in, stereo, curv, inpaint, make_vr180],
        outputs=[preview, sbs_path, tb_path, vr180_path],
    )

# ✅ For direct testing (if run standalone)
if __name__ == "__main__":
    demo.queue(concurrency_count=1).launch(server_name="0.0.0.0", server_port=7860)
