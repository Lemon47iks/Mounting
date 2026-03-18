import re
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

import os
os.environ["PATH"] += os.pathsep + r"C:\Users\Lemon\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin"

import random
import numpy as np
from moviepy.editor import (
    VideoFileClip, ImageClip, AudioFileClip,
    concatenate_videoclips, CompositeVideoClip, TextClip
)
from moviepy.video.tools.subtitles import SubtitlesClip
import whisper

# ──────────────── НАСТРОЙКИ ────────────────
VIDEO_DIR  = "input/videos"
AUDIO_DIR  = "input/audio"
IMAGE_DIR  = "input/images"
OUTPUT     = "output/result.mp4"

RESOLUTION = (1920, 1080)   # финальное разрешение
FPS        = 24
FONT       = "Arial-Bold"   # шрифт субтитров (должен быть в системе)
SUB_SIZE   = 52             # размер субтитров
FADE_DUR   = 0.4            # длительность fade между клипами (сек)

USE_WHISPER    = False       # False = без субтитров
WHISPER_MODEL  = "medium"    # tiny / base / small / medium / large
# ───────────────────────────────────────────


def natural_key(path):
    parts = re.split(r'(\d+)', os.path.basename(path).lower())
    return [int(p) if p.isdigit() else p for p in parts]

def get_files(folder, exts):
    files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder)
         if f.lower().split(".")[-1] in exts],
        key=natural_key
    )
    return files


def resize_clip(clip, target_size):
    """Кроп по центру с сохранением aspect ratio под target_size."""
    tw, th = target_size
    cw, ch = clip.size
    scale = max(tw / cw, th / ch)
    clip = clip.resize(scale)
    x_center = clip.w / 2
    y_center = clip.h / 2
    clip = clip.crop(
        x1=x_center - tw / 2, y1=y_center - th / 2,
        x2=x_center + tw / 2, y2=y_center + th / 2
    )
    return clip


# ── ЭФФЕКТЫ НА КАРТИНКИ ──

def effect_zoom_in(clip):
    def make_frame(t):
        progress = t / clip.duration
        scale = 1.0 + 0.20 * progress  # 1.0 → 1.20 (было 0.05)
        w = int(RESOLUTION[0] * scale)
        h = int(RESOLUTION[1] * scale)
        frame = clip.get_frame(t)
        from PIL import Image
        resized = np.array(Image.fromarray(frame).resize((w, h), Image.LANCZOS))
        y1 = (h - RESOLUTION[1]) // 2
        x1 = (w - RESOLUTION[0]) // 2
        return resized[y1:y1 + RESOLUTION[1], x1:x1 + RESOLUTION[0]]
    return clip.fl(make_frame)

def effect_zoom_out(clip):
    def make_frame(t):
        progress = t / clip.duration
        scale = 1.20 - 0.20 * progress  # 1.20 → 1.0
        w = int(RESOLUTION[0] * scale)
        h = int(RESOLUTION[1] * scale)
        frame = clip.get_frame(t)
        from PIL import Image
        resized = np.array(Image.fromarray(frame).resize((w, h), Image.LANCZOS))
        y1 = (h - RESOLUTION[1]) // 2
        x1 = (w - RESOLUTION[0]) // 2
        return resized[y1:y1 + RESOLUTION[1], x1:x1 + RESOLUTION[0]]
    return clip.fl(make_frame)

def effect_pan_left(clip):
    padding = 200  # было 80, увеличиваем смещение
    def make_frame(t):
        progress = t / clip.duration
        frame = clip.get_frame(t)
        from PIL import Image
        w = RESOLUTION[0] + padding
        resized = np.array(Image.fromarray(frame).resize((w, RESOLUTION[1]), Image.LANCZOS))
        x1 = int(padding * progress)
        return resized[:, x1:x1 + RESOLUTION[0]]
    return clip.fl(make_frame)

def effect_pan_right(clip):
    padding = 200
    def make_frame(t):
        progress = t / clip.duration
        frame = clip.get_frame(t)
        from PIL import Image
        w = RESOLUTION[0] + padding
        resized = np.array(Image.fromarray(frame).resize((w, RESOLUTION[1]), Image.LANCZOS))
        x1 = int(padding * (1 - progress))
        return resized[:, x1:x1 + RESOLUTION[0]]
    return clip.fl(make_frame)


EFFECTS = [effect_zoom_in, effect_zoom_out, effect_pan_left, effect_pan_right]


def apply_random_effect(clip):
    effect_fn = random.choice(EFFECTS)
    try:
        return effect_fn(clip)
    except Exception:
        return clip  # если что-то пошло не так — без эффекта


# ── СУБТИТРЫ ЧЕРЕЗ WHISPER ──

def generate_subtitles(audio_path):
    print("⏳ Генерация субтитров через Whisper...")
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(audio_path, language="ru")
    subs = []
    for seg in result["segments"]:
        subs.append(((seg["start"], seg["end"]), seg["text"].strip()))
    print(f"✅ Субтитров сгенерировано: {len(subs)}")
    return subs


def make_subtitle_clip(subs, video_size):
    w, h = video_size
    def generator(txt):
        return TextClip(
            txt,
            font=FONT,
            fontsize=SUB_SIZE,
            color="white",
            stroke_color="black",
            stroke_width=2,
            size=(int(w * 0.85), None),
            method="caption",
            align="center",
        )
    return SubtitlesClip(subs, generator).set_pos(("center", h - SUB_SIZE * 4))


# ── ГЛАВНАЯ ФУНКЦИЯ ──

def build_video():
    # 1. Загружаем аудио
    audio_files = get_files(AUDIO_DIR, ["mp3", "wav", "m4a", "aac"])
    assert audio_files, "❌ Нет аудиофайлов в input/audio/"
    audio = AudioFileClip(audio_files[0])
    audio_duration = audio.duration
    print(f"🎵 Аудио: {audio_duration:.1f}s ({audio_duration/60:.1f} мин) — {audio_files[0]}")

    # 2. Загружаем видеофрагменты
    video_files = get_files(VIDEO_DIR, ["mp4", "mov", "avi", "mkv"])
    print("Найденные видео файлы:")
    for f in video_files:
        print(" ", f)

    video_clips = []
    video_total_duration = 0
    for vf in video_files:
        vc = VideoFileClip(vf).without_audio()
        vc = resize_clip(vc, RESOLUTION)
        vc = vc.resize(1.15)
        vc = vc.crop(
            x_center=vc.w / 2,
            y_center=vc.h / 2,
            width=RESOLUTION[0],
            height=RESOLUTION[1]
        )
        vc = vc.fadein(FADE_DUR).fadeout(FADE_DUR)
        video_clips.append(vc)
        video_total_duration += vc.duration
    print(f"🎬 Видео: {len(video_clips)} клипов, суммарно {video_total_duration:.1f}s")

    # 3. Рассчитываем длительность каждой картинки
    image_files = get_files(IMAGE_DIR, ["jpg", "jpeg", "png", "webp"])
    assert image_files, "❌ Нет изображений в input/images/"
    n_images = len(image_files)
    images_total_duration = audio_duration - video_total_duration
    if images_total_duration <= 0:
        print("⚠️ Видео длиннее аудио. Картинки будут по 5 сек.")
        images_total_duration = n_images * 5
    image_duration = images_total_duration / n_images
    print(f"🖼  Картинок: {n_images} | Под картинки: {images_total_duration:.1f}s | На каждую: {image_duration:.2f}s")

    # 4. Создаём клипы из картинок с эффектами
    image_clips = []
    for img_path in image_files:
        clip = (ImageClip(img_path)
                .set_duration(image_duration)
                .set_fps(FPS))
        clip = resize_clip(clip, RESOLUTION)
        clip = apply_random_effect(clip)
        clip = clip.fadein(FADE_DUR).fadeout(FADE_DUR)
        image_clips.append(clip)
    print("✅ Картинки обработаны")

    # 5. Склейка: видео → картинки
    all_clips = video_clips + image_clips
    final_video = concatenate_videoclips(all_clips, method="compose")
    print(f"📐 Финальная длина видеоряда: {final_video.duration:.1f}s")

    # 6. Накладываем аудио (аудио — мастер по длине)
    # Обрезаем или дополняем видеоряд до длины аудио
    if final_video.duration < audio_duration:
        # Добавляем чёрный хвост если нужно
        from moviepy.editor import ColorClip
        tail = ColorClip(RESOLUTION, color=(0, 0, 0),
                         duration=audio_duration - final_video.duration).set_fps(FPS)
        final_video = concatenate_videoclips([final_video, tail])
    else:
        final_video = final_video.subclip(0, audio_duration)

    final_video = final_video.set_audio(audio)

    # 7. Субтитры
    if USE_WHISPER:
        subs = generate_subtitles(audio_files[0])
        if subs:
            subtitle_clip = make_subtitle_clip(subs, RESOLUTION)
            final_video = CompositeVideoClip([final_video, subtitle_clip])

    # 8. Рендер
    os.makedirs("output", exist_ok=True)
    print("🚀 Рендеринг...")
    final_video.write_videofile(
        OUTPUT,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset="fast",       # slow/medium/fast — качество vs скорость
        ffmpeg_params=["-crf", "18"],  # 18=высокое качество, 23=среднее
    )
    print(f"✅ Готово! Сохранено: {OUTPUT}")

    # Закрываем клипы
    audio.close()
    for c in all_clips:
        c.close()


if __name__ == "__main__":
    build_video()