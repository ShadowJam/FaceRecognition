import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import shutil
import requests
from io import BytesIO
from typing import List, Optional
import time


def download_sample_faces(output_dir: Path, num_people: int = 5, images_per_person: int = 3) -> bool:
    """Загрузка тестовых изображений лиц из открытых источников"""
    print("Downloading sample faces from open sources...")

    # Используем бесплатный API для тестовых изображений
    base_urls = [
        "https://thispersondoesnotexist.com/image",
        "https://picsum.photos/300/300?person",
        "https://randomuser.me/api/portraits"
    ]

    try:
        for person_id in range(1, num_people + 1):
            person_dir = output_dir / f"person_{person_id}"
            os.makedirs(person_dir, exist_ok=True)

            for img_num in range(1, images_per_person + 1):
                img_path = person_dir / f"face_{img_num}.jpg"

                # Пробуем разные источники
                success = False
                for source_idx, url in enumerate(base_urls):
                    try:
                        if source_idx == 0:  # thispersondoesnotexist.com
                            response = requests.get(url, headers={'User-Agent': 'My-Test-App'})
                            if response.status_code == 200:
                                img = Image.open(BytesIO(response.content))
                                img.save(img_path)
                                success = True
                                break

                        elif source_idx == 1:  # picsum.photos
                            response = requests.get(url)
                            if response.status_code == 200:
                                img = Image.open(BytesIO(response.content))
                                img.save(img_path)
                                success = True
                                break

                        elif source_idx == 2:  # randomuser.me
                            gender = "women" if person_id % 2 == 0 else "men"
                            url = f"{url}/{gender}/{person_id}.jpg"
                            response = requests.get(url)
                            if response.status_code == 200:
                                img = Image.open(BytesIO(response.content))
                                img.save(img_path)
                                success = True
                                break

                    except Exception as e:
                        print(f"Error downloading from {url}: {e}")
                        continue

                if not success:
                    # Fallback: генерируем простое изображение
                    img = Image.new('RGB', (256, 256), color=(73, 109, 137))
                    d = ImageDraw.Draw(img)
                    d.text((80, 100), f"Face {person_id}-{img_num}", fill=(255, 255, 0))
                    img.save(img_path)
                    print(f"Used fallback image for person {person_id}")

                # Добавляем небольшую задержку между запросами
                time.sleep(1)

        return True

    except Exception as e:
        print(f"Failed to download faces: {e}")
        return False


def generate_sample_video(output_path: Path, duration_sec: int = 5, fps: int = 20) -> bool:
    """Генерация тестового видео с лицами"""
    print("Generating sample video...")

    try:
        width, height = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not out.isOpened():
            raise Exception("Failed to open video writer")

        # Генерируем кадры с простой анимацией
        for i in range(fps * duration_sec):
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Рисуем движущийся круг как лицо
            center_x = int(width / 2 + (width / 4) * np.sin(i / fps))
            center_y = int(height / 2 + (height / 4) * np.cos(i / fps))
            cv2.circle(frame, (center_x, center_y), 50, (0, 255, 255), -1)

            # Глаза
            cv2.circle(frame, (center_x - 20, center_y - 10), 10, (0, 0, 0), -1)
            cv2.circle(frame, (center_x + 20, center_y - 10), 10, (0, 0, 0), -1)

            # Рот
            cv2.ellipse(frame, (center_x, center_y + 20),
                        (30, 15), 0, 0, 180, (0, 0, 0), 2)

            out.write(frame)

        out.release()
        return True

    except Exception as e:
        print(f"Video generation failed: {e}")
        return False


def generate_sample_data() -> None:
    """Генерация всех тестовых данных"""
    print("Starting sample data generation...")

    # 1. Создаем структуру папок
    SAMPLE_DATA_DIR = Path(__file__).parent / "sample_data"
    KNOWN_FACES_DIR = SAMPLE_DATA_DIR / "known_faces"
    VIDEOS_DIR = SAMPLE_DATA_DIR / "videos"

    shutil.rmtree(SAMPLE_DATA_DIR, ignore_errors=True)
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    # 2. Загружаем тестовые лица
    download_success = download_sample_faces(KNOWN_FACES_DIR)

    if not download_success:
        print("Using locally generated faces instead")
        # Fallback: генерируем простые изображения
        for person_id in range(1, 6):
            person_dir = KNOWN_FACES_DIR / f"person_{person_id}"
            os.makedirs(person_dir, exist_ok=True)

            for img_num in range(1, 4):
                img = Image.new('RGB', (256, 256), color=(73, 109, 137))
                d = ImageDraw.Draw(img)
                d.text((80, 100), f"Face {person_id}-{img_num}", fill=(255, 255, 0))
                img.save(person_dir / f"face_{img_num}.jpg")

    # 3. Генерируем тестовое видео
    video_path = VIDEOS_DIR / "sample_video.mp4"
    video_success = generate_sample_video(video_path)

    if not video_success:
        print("Using simple image sequence as fallback")
        frame_dir = VIDEOS_DIR / "video_frames"
        os.makedirs(frame_dir, exist_ok=True)

        for i in range(20 * 5):  # 5 секунд при 20 fps
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(frame_dir / f"frame_{i:04d}.jpg"), frame)

    print("Sample data generation completed!")


if __name__ == "__main__":
    generate_sample_data()