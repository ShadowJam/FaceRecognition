import os
from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).parent
    UPLOAD_FOLDER = BASE_DIR / 'uploads/video'
    KNOWN_FACES_FOLDER = BASE_DIR / 'uploads/photo'
    LOGS_FOLDER = BASE_DIR / 'logs'
    STATIC_FOLDER = BASE_DIR / 'static'
    #SAMPLE_DATA_DIR = BASE_DIR / 'sample_data'

    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'}
    VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

    # Параметры анализа
    FRAMES_PER_SECOND = 2
    MIN_CONFIDENCE = 0.3
    FACE_IMG_SIZE = (160, 160)  # Размер для FaceNet

    # Настройки трекера
    TRACKING_DISTANCE_THRESHOLD = 100
    TRACKING_INIT_DELAY = 3
    TRACKING_HIT_COUNTER_MAX = 10
    TRACKING_POINTS_COUNT = 4  # Центр + 2 угла

    @staticmethod
    def init_app(app):
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.KNOWN_FACES_FOLDER, exist_ok=True)
        os.makedirs(Config.LOGS_FOLDER, exist_ok=True)
        #os.makedirs(Config.SAMPLE_DATA_DIR, exist_ok=True)

        # Конфигурация Flask
        app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH