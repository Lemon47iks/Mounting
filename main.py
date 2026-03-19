#!/usr/bin/env python3
"""
Video Montage Queue
-------------------
Ставь сколько угодно роликов в очередь — обработаются по порядку.

Структура папок:
  projects/
  ├── 01_ролик_бухгалтерия/
  │   ├── videos/
  │   ├── images/
  │   └── audio/
  ├── 02_ролик_налоги/
  │   ├── videos/
  │   ├── images/
  │   └── audio/
  └── ...

queue.json — список проектов (создаётся автоматически, можно редактировать):
  [
    { "name": "Ролик 1", "folder": "projects/01_ролик_бухгалтерия" },
    { "name": "Ролик 2", "folder": "projects/02_ролик_налоги",
      "video_scale": 1.20, "zoom_ratio": 0.03 }
  ]

Запуск:   python montage.py
Опции:
  --scan    автоматически найти все папки в projects/ и обновить queue.json
  --list    показать очередь и статус каждого проекта без рендера

Install: pip install "moviepy==1.0.3" Pillow numpy
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    VideoClip,
    VideoFileClip,
    concatenate_videoclips,
)
from moviepy.video.fx.all import crop as fx_crop

# ─── ГЛОБАЛЬНЫЕ НАСТРОЙКИ (применяются если не переопределены в queue.json) ──

PROJECTS_DIR = "projects"    # папка со всеми проектами
QUEUE_FILE   = "queue.json"  # файл очереди
LOG_FILE     = "montage.log" # лог (удобно смотреть утром)

TARGET_W     = 1920
TARGET_H     = 1080
FPS          = 30
VIDEO_SCALE  = 1.15   # масштаб видео для скрытия водяного знака
ZOOM_RATIO   = 0.04   # скорость зума картинок (4% в секунду)

# ─── ЛОГИРОВАНИЕ ──────────────────────────────────────────────────────────────

def setup_logging():
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

log = logging.getLogger(__name__)

# ─── УТИЛИТЫ ──────────────────────────────────────────────────────────────────

def natural_sort_key(name: str):
    m = re.match(r"^(\d+)", name)
    return int(m.group(1)) if m else name

def collect_files(folder: str, extensions: tuple) -> list:
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"Папка не найдена: '{folder}'")
    files = [f.name for f in p.iterdir() if f.suffix.lower() in extensions]
    if not files:
        raise FileNotFoundError(f"Нет файлов в '{folder}'")
    return sorted(files, key=natural_sort_key)

def fmt_duration(seconds: float) -> str:
    """Форматирует секунды → '1ч 23м 45с'"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    parts = []
    if h: parts.append(f"{h}ч")
    if m: parts.append(f"{m}м")
    parts.append(f"{s}с")
    return " ".join(parts)

def fmt_eta(seconds: float) -> str:
    eta = datetime.now() + timedelta(seconds=seconds)
    return eta.strftime("%H:%M")

# ─── ОБРАБОТКА ВИДЕО ──────────────────────────────────────────────────────────

def scale_and_crop_video(clip, scale: float, target_w: int, target_h: int):
    orig_w, orig_h = clip.size
    cover = max(target_w / orig_w, target_h / orig_h) * scale
    new_w = int(orig_w * cover)
    new_h = int(orig_h * cover)
    scaled = clip.resize(newsize=(new_w, new_h))
    x1 = (new_w - target_w) // 2
    y1 = (new_h - target_h) // 2
    return fx_crop(scaled, x1=x1, y1=y1, x2=x1 + target_w, y2=y1 + target_h)

# ─── ОБРАБОТКА КАРТИНОК ───────────────────────────────────────────────────────

def _cover_crop(image_path: str, target_w: int, target_h: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    iw, ih = img.size
    scale = max(target_w / iw, target_h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    x = (nw - target_w) // 2
    y = (nh - target_h) // 2
    return np.array(img.crop([x, y, x + target_w, y + target_h]))

def make_image_clip(image_path: str, duration: float,
                    zoom_ratio: float, target_w: int, target_h: int) -> VideoClip:
    """
    Плавный Ken Burns зум без тряски.
    Принцип: один раз масштабируем до максимального зума,
    затем на каждом кадре только crop (нет resize в цикле → нет тряски).
    """
    max_zoom = 1 + zoom_ratio * duration

    base_arr = _cover_crop(image_path, target_w, target_h)
    bh, bw = base_arr.shape[:2]
    big_w = int(bw * max_zoom)
    big_h = int(bh * max_zoom)
    big_img = np.array(
        Image.fromarray(base_arr).resize((big_w, big_h), Image.LANCZOS)
    )

    def make_frame(t: float) -> np.ndarray:
        current_zoom = 1 + zoom_ratio * t
        win_w = int(bw * max_zoom / current_zoom)
        win_h = int(bh * max_zoom / current_zoom)
        x = (big_w - win_w) // 2
        y = (big_h - win_h) // 2
        crop_arr = big_img[y:y + win_h, x:x + win_w]
        return np.array(
            Image.fromarray(crop_arr).resize((target_w, target_h), Image.LANCZOS)
        )

    return VideoClip(make_frame, duration=duration).set_fps(FPS)

# ─── РЕНДЕР ОДНОГО ПРОЕКТА ────────────────────────────────────────────────────

def render_project(cfg: dict) -> bool:
    """
    Рендерит один проект. Возвращает True если успешно, False если ошибка.
    cfg: {
        name, folder,
        output (опц),
        target_w, target_h, fps,
        video_scale, zoom_ratio
    }
    """
    name       = cfg.get("name", cfg["folder"])
    folder     = Path(cfg["folder"])
    output     = cfg.get("output", str(folder / "output.mp4"))
    target_w   = cfg.get("target_w",    TARGET_W)
    target_h   = cfg.get("target_h",    TARGET_H)
    fps        = cfg.get("fps",         FPS)
    v_scale    = cfg.get("video_scale", VIDEO_SCALE)
    zoom_ratio = cfg.get("zoom_ratio",  ZOOM_RATIO)

    log.info("=" * 60)
    log.info(f"▶  ПРОЕКТ: {name}")
    log.info(f"   Папка:  {folder}")
    log.info(f"   Выход:  {output}")
    log.info("=" * 60)

    t_start = time.time()

    try:
        # 1. Аудио
        audio_exts = (".mp3", ".wav", ".aac", ".m4a", ".ogg", ".flac")
        audio_files = collect_files(str(folder / "audio"), audio_exts)
        audio_clip = AudioFileClip(str(folder / "audio" / audio_files[0]))
        total_audio = audio_clip.duration
        log.info(f"🎵  Аудио: {audio_files[0]}  ({fmt_duration(total_audio)})")

        # 2. Видеоклипы
        video_exts  = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
        video_names = collect_files(str(folder / "videos"), video_exts)
        log.info(f"🎬  Видео: {len(video_names)} файлов")

        video_clips = []
        total_video_dur = 0.0
        for vname in video_names:
            raw  = VideoFileClip(str(folder / "videos" / vname))
            clip = scale_and_crop_video(raw, v_scale, target_w, target_h).set_fps(fps)
            video_clips.append(clip)
            total_video_dur += clip.duration
            log.info(f"    ✓  {vname}  ({clip.duration:.1f}с)")
        log.info(f"    Итого видео: {fmt_duration(total_video_dur)}")

        # 3. Картинки
        image_exts  = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif")
        image_names = collect_files(str(folder / "images"), image_exts)

        images_dur = total_audio - total_video_dur
        if images_dur <= 0:
            raise ValueError("Видео длиннее или равно аудио — нет времени для картинок.")

        per_image = images_dur / len(image_names)
        log.info(f"🖼️   Картинки: {len(image_names)} шт.")
        log.info(f"    Аудио({fmt_duration(total_audio)}) − Видео({fmt_duration(total_video_dur)})"
                 f" = {fmt_duration(images_dur)} ÷ {len(image_names)} = {per_image:.2f}с/кадр")

        image_clips = []
        for i, iname in enumerate(image_names, 1):
            print(f"    [{i:>4}/{len(image_names)}] {iname}", end="\r", flush=True)
            image_clips.append(
                make_image_clip(str(folder / "images" / iname),
                                per_image, zoom_ratio, target_w, target_h)
            )
        print()
        log.info(f"    ✅  {len(image_clips)} клипов готово")

        # 4. Сборка
        log.info("⚙️   Сборка и рендер …")
        final = concatenate_videoclips(video_clips + image_clips, method="compose")
        final = final.set_audio(audio_clip)
        log.info(f"    Итоговая длина: {fmt_duration(final.duration)}")

        # 5. Рендер
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        final.write_videofile(
            output,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            threads=os.cpu_count(),
            preset="fast",
            logger="bar",
        )

        elapsed = time.time() - t_start
        log.info(f"✅  ГОТОВО: {name}  (затрачено: {fmt_duration(elapsed)})")
        log.info(f"   Файл: {output}")
        return True

    except Exception as e:
        elapsed = time.time() - t_start
        log.error(f"❌  ОШИБКА в проекте '{name}': {e}")
        log.error(f"   Пропускаем, переходим к следующему…")
        return False

# ─── ОЧЕРЕДЬ ──────────────────────────────────────────────────────────────────

def load_queue() -> list:
    """Загружает queue.json. Поддерживает строки (путь) и объекты (dict)."""
    if not Path(QUEUE_FILE).exists():
        log.warning(f"queue.json не найден. Запусти с --scan для автогенерации.")
        return []

    with open(QUEUE_FILE, encoding="utf-8") as f:
        raw = json.load(f)

    queue = []
    for item in raw:
        if isinstance(item, str):
            queue.append({"folder": item, "name": Path(item).name})
        elif isinstance(item, dict):
            if "folder" not in item:
                log.warning(f"Пропущен элемент без поля 'folder': {item}")
                continue
            item.setdefault("name", Path(item["folder"]).name)
            queue.append(item)
    return queue

def scan_projects() -> None:
    """Находит все подпапки в PROJECTS_DIR и записывает в queue.json."""
    p = Path(PROJECTS_DIR)
    if not p.exists():
        p.mkdir(parents=True)
        log.info(f"Создана папка {PROJECTS_DIR}/")

    folders = sorted(
        [str(sub) for sub in p.iterdir() if sub.is_dir()],
        key=lambda x: natural_sort_key(Path(x).name)
    )

    if not folders:
        log.info(f"Папок в {PROJECTS_DIR}/ не найдено.")
        return

    # Читаем существующий queue.json чтобы сохранить кастомные настройки
    existing = {}
    if Path(QUEUE_FILE).exists():
        with open(QUEUE_FILE, encoding="utf-8") as f:
            for item in json.load(f):
                if isinstance(item, dict):
                    existing[item.get("folder")] = item

    queue = []
    for folder in folders:
        if folder in existing:
            queue.append(existing[folder])   # сохраняем кастомные настройки
        else:
            queue.append({"name": Path(folder).name, "folder": folder})

    with open(QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(queue, f, ensure_ascii=False, indent=2)

    log.info(f"queue.json обновлён: {len(queue)} проектов")
    for item in queue:
        log.info(f"  • {item['name']}  ({item['folder']})")

def show_queue(queue: list) -> None:
    """Показывает статус очереди без рендера."""
    print(f"\n{'─'*58}")
    print(f"  ОЧЕРЕДЬ: {len(queue)} проектов")
    print(f"{'─'*58}")
    for i, cfg in enumerate(queue, 1):
        folder = Path(cfg["folder"])
        output = cfg.get("output", str(folder / "output.mp4"))
        done   = "✅ готово  " if Path(output).exists() else "⏳ ожидает"
        print(f"  {i:>2}. [{done}]  {cfg['name']}")
        print(f"       📁 {folder}")
    print(f"{'─'*58}\n")

# ─── ТОЧКА ВХОДА ──────────────────────────────────────────────────────────────

def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Video Montage Queue")
    parser.add_argument("--scan",  action="store_true",
                        help="Найти проекты в projects/ и обновить queue.json")
    parser.add_argument("--list",  action="store_true",
                        help="Показать очередь без рендера")
    parser.add_argument("--force", action="store_true",
                        help="Рендерить даже если output.mp4 уже существует")
    args = parser.parse_args()

    if args.scan:
        scan_projects()
        if not args.list:
            return

    queue = load_queue()
    if not queue:
        return

    if args.list:
        show_queue(queue)
        return

    # ── Запуск очереди ────────────────────────────────────────────────────────
    total   = len(queue)
    results = {"ok": [], "skip": [], "fail": []}
    t_all   = time.time()

    log.info(f"\n🚀  Запуск очереди: {total} проектов")
    log.info(f"    Время старта: {datetime.now().strftime('%H:%M:%S')}\n")

    for idx, cfg in enumerate(queue, 1):
        output = cfg.get("output", str(Path(cfg["folder"]) / "output.mp4"))

        # Пропускаем уже готовые (если нет --force)
        if Path(output).exists() and not args.force:
            log.info(f"[{idx}/{total}] ⏭  Пропускаем (уже есть): {cfg['name']}")
            results["skip"].append(cfg["name"])
            continue

        log.info(f"\n[{idx}/{total}] Начинаем: {cfg['name']}")

        ok = render_project(cfg)
        if ok:
            results["ok"].append(cfg["name"])
        else:
            results["fail"].append(cfg["name"])

    # ── Итоговый отчёт ────────────────────────────────────────────────────────
    elapsed = time.time() - t_all
    log.info("\n" + "=" * 60)
    log.info("  ИТОГ ОЧЕРЕДИ")
    log.info("=" * 60)
    log.info(f"  Общее время:  {fmt_duration(elapsed)}")
    log.info(f"  ✅ Готово:    {len(results['ok'])}  — {', '.join(results['ok']) or '—'}")
    log.info(f"  ⏭  Пропущено: {len(results['skip'])}  — {', '.join(results['skip']) or '—'}")
    log.info(f"  ❌ Ошибки:    {len(results['fail'])}  — {', '.join(results['fail']) or '—'}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
