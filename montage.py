#!/usr/bin/env python3
"""
Video Montage Tool
------------------
Combines video clips + images (with Ken Burns zoom) + voiceover into one file.

Folder structure:
  videos/  — video clips  (sorted by leading number: 1_..., 2_..., ...)
  images/  — image files  (sorted by leading number: 1_..., 2_..., ...)
  audio/   — one audio file (voiceover)

Install: pip install "moviepy==1.0.3" Pillow numpy
Run:     python montage.py
"""

import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
)
from moviepy.video.fx.all import crop as fx_crop

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

VIDEO_FOLDER = "videos"    # folder with video clips
AUDIO_FOLDER = "audio"     # folder with voiceover
IMAGE_FOLDER = "images"    # folder with images
OUTPUT_FILE  = "output.mp4"

TARGET_W = 1920    # output width  (px)
TARGET_H = 1080    # output height (px)
FPS      = 30      # output frame rate

# Videos are scaled up by this factor to hide corner watermarks
# (scaled then center-cropped back — no black bars)
VIDEO_SCALE = 1.15

# Ken Burns zoom speed for images: 0.04 = 4% per second
# Example: 10-second image zooms from 100 % → 140 %
ZOOM_RATIO = 0.04

# ──────────────────────────────────────────────────────────────────────────────


def natural_sort_key(name: str):
    """Sort files by their leading integer prefix (e.g. '2_1_name' → 2)."""
    m = re.match(r"^(\d+)", name)
    return int(m.group(1)) if m else name


def collect_files(folder: str, extensions: tuple) -> list:
    """Return files in `folder` matching `extensions`, sorted naturally."""
    p = Path(folder)
    if not p.exists():
        sys.exit(f"❌  Folder not found: '{folder}'")
    files = [f.name for f in p.iterdir() if f.suffix.lower() in extensions]
    if not files:
        sys.exit(f"❌  No matching files in '{folder}'")
    return sorted(files, key=natural_sort_key)


# ── Video helpers ─────────────────────────────────────────────────────────────

def scale_and_crop_video(clip, scale: float = VIDEO_SCALE):
    """
    Scale video up and center-crop to TARGET_W × TARGET_H.
    Hides watermarks at edges; guarantees no black bars.
    """
    orig_w, orig_h = clip.size
    # Scale enough to cover target at the requested ratio
    cover = max(TARGET_W / orig_w, TARGET_H / orig_h) * scale
    new_w = int(orig_w * cover)
    new_h = int(orig_h * cover)

    scaled = clip.resize(newsize=(new_w, new_h))
    x1 = (new_w - TARGET_W) // 2
    y1 = (new_h - TARGET_H) // 2
    return fx_crop(scaled, x1=x1, y1=y1, x2=x1 + TARGET_W, y2=y1 + TARGET_H)


# ── Image helpers ─────────────────────────────────────────────────────────────

def _cover_crop(image_path: str) -> np.ndarray:
    """
    Open image → scale to *cover* TARGET_W × TARGET_H (no letterbox) →
    center-crop to exact target → return numpy array (H, W, 3).
    Images are never stretched beyond their natural proportions.
    """
    img = Image.open(image_path).convert("RGB")
    iw, ih = img.size
    scale = max(TARGET_W / iw, TARGET_H / ih)   # cover, not contain
    nw, nh = int(iw * scale), int(ih * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    x = (nw - TARGET_W) // 2
    y = (nh - TARGET_H) // 2
    return np.array(img.crop([x, y, x + TARGET_W, y + TARGET_H]))


def make_image_clip(image_path: str, duration: float,
                    zoom_ratio: float = ZOOM_RATIO) -> ImageClip:
    """
    Create a video clip from an image with smooth zoom-in (Ken Burns) effect.

    How it works:
      • Image is pre-scaled to exactly cover the output frame (no black).
      • Each frame t: scale up by (1 + zoom_ratio * t), then center-crop back
        to TARGET size → progressive zoom-in, never reveals black borders.
    """
    base_arr = _cover_crop(image_path)
    clip = ImageClip(base_arr).set_duration(duration).set_fps(FPS)

    def zoom_effect(get_frame, t: float) -> np.ndarray:
        frame = get_frame(t)
        pil = Image.fromarray(frame)
        bw, bh = pil.size
        nw = math.ceil(bw * (1 + zoom_ratio * t))
        nh = math.ceil(bh * (1 + zoom_ratio * t))
        pil = pil.resize((nw, nh), Image.LANCZOS)
        x = (nw - bw) // 2
        y = (nh - bh) // 2
        return np.array(pil.crop([x, y, x + bw, y + bh]))

    return clip.fl(zoom_effect)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:

    # ── 1. Audio ──────────────────────────────────────────────────────────────
    audio_exts = (".mp3", ".wav", ".aac", ".m4a", ".ogg", ".flac")
    audio_files = collect_files(AUDIO_FOLDER, audio_exts)

    audio_clip = AudioFileClip(os.path.join(AUDIO_FOLDER, audio_files[0]))
    total_audio = audio_clip.duration
    print(f"🎵  Audio : {audio_files[0]}")
    print(f"    Duration: {total_audio:.1f}s  ({total_audio / 60:.2f} min)")

    # ── 2. Video clips ────────────────────────────────────────────────────────
    video_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
    video_names = collect_files(VIDEO_FOLDER, video_exts)

    print(f"\n🎬  Processing {len(video_names)} video clips …")
    video_clips = []
    total_video_dur = 0.0

    for vname in video_names:
        raw  = VideoFileClip(os.path.join(VIDEO_FOLDER, vname))
        clip = scale_and_crop_video(raw, VIDEO_SCALE).set_fps(FPS)
        video_clips.append(clip)
        total_video_dur += clip.duration
        print(f"    ✓  {vname}  ({clip.duration:.1f}s)")

    print(f"    → Total video: {total_video_dur:.1f}s ({total_video_dur / 60:.2f} min)")

    # ── 3. Images  ────────────────────────────────────────────────────────────
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif")
    image_names = collect_files(IMAGE_FOLDER, image_exts)

    images_dur = total_audio - total_video_dur
    if images_dur <= 0:
        sys.exit("❌  Video duration already ≥ audio — no time left for images.")

    per_image = images_dur / len(image_names)

    print(f"\n🖼️   Images : {len(image_names)} files")
    print(f"    Audio({total_audio:.0f}s) − Video({total_video_dur:.0f}s)"
          f" = {images_dur:.0f}s  ÷  {len(image_names)} images"
          f" = {per_image:.2f}s / image")

    print("\n    Building image clips …")
    image_clips = []
    for i, iname in enumerate(image_names, 1):
        print(f"    [{i:>4} / {len(image_names)}] {iname}", end="\r", flush=True)
        image_clips.append(
            make_image_clip(os.path.join(IMAGE_FOLDER, iname), per_image)
        )
    print(f"\n    ✅  {len(image_clips)} image clips ready")

    # ── 4. Assemble & render ──────────────────────────────────────────────────
    print("\n⚙️   Concatenating all clips …")
    final = concatenate_videoclips(video_clips + image_clips, method="compose")
    final = final.set_audio(audio_clip)

    print(f"    Total output duration: {final.duration:.1f}s ({final.duration / 60:.2f} min)")
    print(f"\n💾  Rendering → {OUTPUT_FILE} …")

    final.write_videofile(
        OUTPUT_FILE,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        threads=os.cpu_count(),
        preset="fast",
        logger="bar",
    )

    print(f"\n✅  Done!  →  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
