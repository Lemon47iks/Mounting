import os
import re
from pathlib import Path
from typing import List, Tuple

from PIL import Image

# Совместимость со старым MoviePy + новым Pillow
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

from moviepy.editor import (
    AudioFileClip,
    VideoFileClip,
    ImageClip,
    CompositeVideoClip,
    concatenate_videoclips,
)


# =========================
# НАСТРОЙКИ
# =========================
BASE_DIR = Path(__file__).resolve().parent


VIDEO_EXTRA_ZOOM = 1.15   # 115% для видео
IMAGE_EXTRA_ZOOM = 1.12   # безопасный запас для картинок
IMAGE_MAX_SHIFT_X = 80    # безопасное смещение по X
IMAGE_MAX_SHIFT_Y = 45    # безопасное смещение по Y
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
    Подгоняет клип под размер кадра без черных рамок.
    Обрезает лишнее по центру.
    """
    target_w, target_h = target_size
    clip_w, clip_h = clip.size

    scale = max(target_w / clip_w, target_h / clip_h)
    resized = clip.resize(scale)

    x_center = resized.w / 2
    y_center = resized.h / 2

    cropped = resized.crop(
        x_center=x_center,
        y_center=y_center,
        width=target_w,
        height=target_h,
    )
    return cropped


def fit_video_clip_to_frame(clip, target_size: Tuple[int, int], extra_zoom: float = 1.15):
    """
    Подгоняет видео под кадр, а потом дополнительно увеличивает,
    чтобы скрыть водяные знаки/края.
    """
    fitted = fit_clip_to_frame(clip, target_size)

    zoomed = fitted.resize(extra_zoom)

    target_w, target_h = target_size
    x_center = zoomed.w / 2
    y_center = zoomed.h / 2

    cropped = zoomed.crop(
        x_center=x_center,
        y_center=y_center,
        width=target_w,
        height=target_h,
    )
    return cropped


def fit_image_clip_to_frame(clip, target_size: Tuple[int, int], extra_zoom: float = 1.12):
    """
    Подгоняет картинку под кадр и даёт небольшой запас,
    чтобы можно было делать Ken Burns без черных полей.
    """
    fitted = fit_clip_to_frame(clip, target_size)
    zoomed = fitted.resize(extra_zoom)
    return zoomed


def make_ken_burns_image_clip(
    image_path,
    duration,
    target_size,
    motion_index,
):
    """
    ПЛАВНЫЙ ЭФФЕКТ БЕЗ ТРЯСКИ:
    - только zoom
    - без движения
    """

    base = ImageClip(str(image_path)).set_duration(duration)
    base = fit_image_clip_to_frame(base, target_size, extra_zoom=IMAGE_EXTRA_ZOOM)

    # варианты только zoom (без движения)
    zooms = [
        (1.00, 1.06),  # лёгкий zoom in
        (1.02, 1.08),
        (1.05, 1.00),  # zoom out
    ]

    zoom_start, zoom_end = zooms[motion_index % len(zooms)]

    def scale_func(t):
        progress = min(max(t / duration, 0), 1)
        return zoom_start + (zoom_end - zoom_start) * progress

    # ВАЖНО: всегда центр
    animated = base.resize(scale_func).set_position(("center", "center"))

    final_clip = CompositeVideoClip(
        [animated],
        size=target_size,
    ).set_duration(duration)

    return final_clip


def build_video_clips(video_paths: List[Path], target_size: Tuple[int, int]):
    clips = []
    total_video_duration = 0.0

    for path in video_paths:
        clip = VideoFileClip(str(path))
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

        final_video = concatenate_videoclips(sequence, method="compose")
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
            preset="medium",
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