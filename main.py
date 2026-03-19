import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from moviepy.editor import (
    AudioFileClip,
    VideoFileClip,
    VideoClip,
    concatenate_videoclips,
)

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS


# =========================
# НАСТРОЙКИ
# =========================
BASE_DIR = Path(__file__).resolve().parent


VIDEO_EXTRA_ZOOM = 1.20   # сильнее приближаем видео, чтобы скрыть края/водяной знак
IMAGE_BASE_EXTRA_ZOOM = 1.18   # стартовый запас для картинок
IMAGE_ZOOM_IN_AMOUNT = 0.08    # постоянный плавный зум на сближение
SHAKE_X = 18                   # тряска по X
SHAKE_Y = 10                   # тряска по Y
SHAKE_FREQ_1 = 0.90
SHAKE_FREQ_2 = 1.70

AUDIO_DIR = BASE_DIR / "assets" / "audio"
VIDEO_DIR = BASE_DIR / "assets" / "videos"
IMAGE_DIR = BASE_DIR / "assets" / "images"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_FILE = OUTPUT_DIR / "final_video.mp4"

TARGET_SIZE = (1920, 1080)
FPS = 30

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}


# =========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================
def natural_sort_key(path: Path):
    """
    Естественная сортировка:
    1_...
    2_...
    10_...
    """
    name = path.stem.lower()
    parts = re.split(r"(\d+)", name)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return key


def list_files(folder: Path, extensions: set) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Папка не найдена: {folder}")

    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions]
    files.sort(key=natural_sort_key)
    return files


def get_single_audio_file(audio_dir: Path) -> Path:
    audio_files = list_files(audio_dir, AUDIO_EXTENSIONS)

    if not audio_files:
        raise FileNotFoundError(f"В папке {audio_dir} не найден аудиофайл")

    if len(audio_files) > 1:
        print("Найдено несколько аудиофайлов. Будет использован первый по сортировке:")
        for file in audio_files:
            print(" -", file.name)

    return audio_files[0]


def fit_clip_to_frame(clip, target_size: Tuple[int, int]):
    """
    Подгоняет клип под кадр без чёрных рамок.
    Лишнее обрезается по центру.
    """
    target_w, target_h = target_size
    clip_w, clip_h = clip.size

    scale = max(target_w / clip_w, target_h / clip_h)
    resized = clip.resize(scale)

    cropped = resized.crop(
        x_center=resized.w / 2,
        y_center=resized.h / 2,
        width=target_w,
        height=target_h,
    )
    return cropped


def fit_video_clip_to_frame(clip, target_size: Tuple[int, int], extra_zoom: float = 1.20):
    """
    Видео:
    1. Подгоняем под кадр
    2. Дополнительно увеличиваем
    3. Снова жёстко обрезаем по центру
    """
    target_w, target_h = target_size

    fitted = fit_clip_to_frame(clip, target_size)
    zoomed = fitted.resize(extra_zoom)

    final_clip = zoomed.crop(
        x_center=zoomed.w / 2,
        y_center=zoomed.h / 2,
        width=target_w,
        height=target_h,
    )

    return final_clip


def make_ken_burns_image_clip(
    image_path: Path,
    duration: float,
    target_size: Tuple[int, int],
    motion_index: int,
):
    """
    Картинка:
    - постоянный плавный зум на приближение
    - тряска сохранена
    - чёрные рамки не появляются
    """
    target_w, target_h = target_size

    pil_image = Image.open(image_path).convert("RGB")
    img_w, img_h = pil_image.size

    # Базовый масштаб: картинка должна с запасом покрывать весь кадр
    base_scale = max(target_w / img_w, target_h / img_h) * IMAGE_BASE_EXTRA_ZOOM

    # Небольшое разнообразие по силе приближения
    zoom_variants = [
        IMAGE_ZOOM_IN_AMOUNT,
        IMAGE_ZOOM_IN_AMOUNT + 0.02,
        IMAGE_ZOOM_IN_AMOUNT + 0.03,
    ]
    zoom_amount = zoom_variants[motion_index % len(zoom_variants)]

    # Небольшое разнообразие по фазе тряски
    phase = motion_index * 0.7

    def make_frame(t):
        progress = min(max(t / duration, 0), 1)

        # Только приближение
        zoom = 1.0 + zoom_amount * progress
        current_scale = base_scale * zoom

        new_w = max(1, int(img_w * current_scale))
        new_h = max(1, int(img_h * current_scale))

        resized = pil_image.resize((new_w, new_h), RESAMPLE_LANCZOS)

        # Сколько реально можно безопасно сдвигать без чёрных краёв
        max_offset_x = max(0, (new_w - target_w) // 2)
        max_offset_y = max(0, (new_h - target_h) // 2)

        # Плавная тряска
        raw_x = int(
            np.sin(2 * np.pi * SHAKE_FREQ_1 * progress + phase) * SHAKE_X
            + np.sin(2 * np.pi * SHAKE_FREQ_2 * progress + phase * 0.5) * (SHAKE_X * 0.35)
        )
        raw_y = int(
            np.cos(2 * np.pi * SHAKE_FREQ_1 * progress + phase) * SHAKE_Y
            + np.cos(2 * np.pi * (SHAKE_FREQ_2 + 0.2) * progress + phase * 0.4) * (SHAKE_Y * 0.35)
        )

        # Ограничиваем тряску безопасными пределами
        offset_x = max(-max_offset_x, min(max_offset_x, raw_x))
        offset_y = max(-max_offset_y, min(max_offset_y, raw_y))

        center_x = new_w // 2 + offset_x
        center_y = new_h // 2 + offset_y

        left = center_x - target_w // 2
        top = center_y - target_h // 2

        # Ещё раз жёстко ограничиваем crop, чтобы не вылезти за края
        left = max(0, min(left, new_w - target_w))
        top = max(0, min(top, new_h - target_h))

        right = left + target_w
        bottom = top + target_h

        cropped = resized.crop((left, top, right, bottom))

        # Защита на случай редких пограничных ситуаций
        if cropped.size != (target_w, target_h):
            background = Image.new("RGB", (target_w, target_h), (0, 0, 0))
            paste_x = max(0, (target_w - cropped.size[0]) // 2)
            paste_y = max(0, (target_h - cropped.size[1]) // 2)
            background.paste(cropped, (paste_x, paste_y))
            cropped = background

        return np.array(cropped)

    clip = VideoClip(make_frame=make_frame, duration=duration)
    clip = clip.set_fps(FPS)
    return clip


def build_video_clips(video_paths: List[Path], target_size: Tuple[int, int]):
    clips = []
    total_video_duration = 0.0

    for path in video_paths:
        clip = VideoFileClip(str(path), audio=False)
        clip = fit_video_clip_to_frame(clip, target_size, extra_zoom=VIDEO_EXTRA_ZOOM)
        clips.append(clip)
        total_video_duration += clip.duration

    return clips, total_video_duration


def build_image_clips(
    image_paths: List[Path],
    images_total_duration: float,
    target_size: Tuple[int, int],
):
    if not image_paths:
        return [], 0.0, 0.0

    duration_per_image = images_total_duration / len(image_paths)

    if duration_per_image <= 0:
        raise ValueError(
            "На картинки не осталось времени. "
            "Проверь длительность озвучки и длительность видео."
        )

    clips = []
    for i, path in enumerate(image_paths):
        clip = make_ken_burns_image_clip(
            image_path=path,
            duration=duration_per_image,
            target_size=target_size,
            motion_index=i,
        )
        clips.append(clip)

    total_images_duration = duration_per_image * len(image_paths)
    return clips, duration_per_image, total_images_duration


def close_clips(clips):
    for clip in clips:
        try:
            clip.close()
        except Exception:
            pass


# =========================
# ОСНОВНАЯ ЛОГИКА
# =========================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audio_path = get_single_audio_file(AUDIO_DIR)
    video_paths = list_files(VIDEO_DIR, VIDEO_EXTENSIONS)
    image_paths = list_files(IMAGE_DIR, IMAGE_EXTENSIONS)

    if not video_paths:
        print("В папке с видео нет файлов.")
    if not image_paths:
        print("В папке с картинками нет файлов.")
    if not video_paths and not image_paths:
        raise RuntimeError("Нет ни видео, ни картинок для сборки ролика.")

    print("\n=== Найденные файлы ===")
    print("Аудио:", audio_path.name)
    print(f"Видео: {len(video_paths)} шт.")
    for p in video_paths:
        print(" -", p.name)

    print(f"Картинки: {len(image_paths)} шт.")
    for p in image_paths[:10]:
        print(" -", p.name)
    if len(image_paths) > 10:
        print(f" ... и ещё {len(image_paths) - 10} картинок")

    audio_clip = AudioFileClip(str(audio_path))
    audio_duration = audio_clip.duration

    print("\n=== Длительности ===")
    print(f"Длина озвучки: {audio_duration:.2f} сек ({audio_duration / 60:.2f} мин)")

    video_clips = []
    image_clips = []
    final_video = None

    try:
        video_clips, total_video_duration = build_video_clips(video_paths, TARGET_SIZE)

        print(
            f"Суммарная длина видеоряда: "
            f"{total_video_duration:.2f} сек ({total_video_duration / 60:.2f} мин)"
        )

        images_total_duration = audio_duration - total_video_duration

        if image_paths and images_total_duration <= 0:
            raise ValueError(
                "Озвучка короче или равна длительности видео. "
                "Для картинок времени не осталось."
            )

        if image_paths:
            image_clips, duration_per_image, total_images_duration = build_image_clips(
                image_paths=image_paths,
                images_total_duration=images_total_duration,
                target_size=TARGET_SIZE,
            )

            print(
                f"Время под картинки: "
                f"{images_total_duration:.2f} сек ({images_total_duration / 60:.2f} мин)"
            )
            print(f"Длительность одной картинки: {duration_per_image:.2f} сек")
        else:
            total_images_duration = 0.0

        sequence = video_clips + image_clips

        final_video = concatenate_videoclips(sequence, method="chain")
        final_video = final_video.set_audio(audio_clip)

        # Если итог из-за округлений чуть длиннее аудио — подрезаем
        if final_video.duration > audio_duration:
            final_video = final_video.subclip(0, audio_duration)

        print("\n=== Экспорт ===")
        print("Сохраняю в:", OUTPUT_FILE)

        final_video.write_videofile(
            str(OUTPUT_FILE),
            fps=FPS,
            codec="libx264",
            audio_codec="aac",
            threads=os.cpu_count() or 4,
            preset="ultrafast",
                )

        print("\nГотово.")
        print(f"Файл сохранён: {OUTPUT_FILE}")

    finally:
        if final_video is not None:
            try:
                final_video.close()
            except Exception:
                pass

        try:
            audio_clip.close()
        except Exception:
            pass

        close_clips(video_clips)
        close_clips(image_clips)


if __name__ == "__main__":
    main()