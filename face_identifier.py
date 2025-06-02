import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
from config import Config
import os
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List


class FaceIdentifier:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device=device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        self.known_faces: Dict[str, torch.Tensor] = {}
        self.load_known_faces()

    def load_known_faces(self) -> None:
        """Загружает известные лица из файла"""
        faces_file = Config.KNOWN_FACES_FOLDER / 'faces.dat'
        try:
            if faces_file.exists():
                with faces_file.open('rb') as f:
                    self.known_faces = pickle.load(f)
                    print(f"Loaded {len(self.known_faces)} known faces from storage")
        except Exception as e:
            print(f"Error loading known faces: {e}")
            self.known_faces = {}

    def save_known_faces(self) -> bool:
        """Сохраняет текущие известные лица в файл"""
        try:
            faces_file = Config.KNOWN_FACES_FOLDER / 'faces.dat'
            os.makedirs(faces_file.parent, exist_ok=True)

            with faces_file.open('wb') as f:
                pickle.dump(self.known_faces, f, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        except Exception as e:
            print(f"Error saving known faces: {e}")
            return False

    def _get_face_embedding(self, img: Image.Image) -> Optional[torch.Tensor]:
        try:
            # Ресайз и конвертация в RGB
            img = img.resize(Config.FACE_IMG_SIZE).convert('RGB')
            face = self.mtcnn(img)

            if face is None:
                return None

            # Нормализация размерности тензора
            if face.dim() == 3:
                face = face.unsqueeze(0)  # Добавляем batch dimension
            elif face.dim() == 4:
                if face.shape[0] > 1:
                    face = face[0].unsqueeze(0)  # Берем первое лицо если несколько

            # Проверка финальной размерности
            if face.dim() != 4 or face.shape[1:] != (3, *Config.FACE_IMG_SIZE):
                raise ValueError(f"Invalid tensor shape: {face.shape}")

            face = face.to(self.device)
            return self.resnet(face).detach().cpu()

        except Exception as e:
            print(f"Face embedding error: {e}")
            return None

    def get_embedding(self, image_path):
        try:
            img = Image.open(image_path)
            # Проверка минимального размера
            if min(img.size) < 128:
                raise ValueError("Image too small for face detection")
            return self._get_face_embedding(img)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def add_person(self, person_id: str, image_paths: List[str]) -> bool:
        """Добавляет нового человека в базу данных"""
        embeddings = []
        for path in image_paths:
            emb = self.get_embedding(path)
            if emb is not None:
                embeddings.append(emb)

        if embeddings:
            # Объединяем все эмбеддинги и вычисляем среднее
            stacked = torch.stack(embeddings)
            self.known_faces[person_id] = stacked.mean(dim=0)
            return self.save_known_faces()
        return False

    def identify_face(self, face_image: Image.Image) -> tuple:
        """Идентифицирует лицо на изображении"""
        try:
            embedding = self._get_face_embedding(face_image)
            if embedding is None:
                return "Unknown", 0.0

            if not self.known_faces:
                return "Unknown", 0.0

            min_dist = float('inf')
            best_match = "Unknown"

            for name, known_emb in self.known_faces.items():
                dist = (embedding - known_emb).norm().item()
                if dist < min_dist:
                    min_dist = dist
                    best_match = name

            confidence = 1 - min(1.0, min_dist / 1.5)

            if confidence >= Config.MIN_CONFIDENCE:
                return best_match, confidence
            return "Unknown", confidence
        except Exception as e:
            print(f"Face identification error: {e}")
            return "Unknown", 0.0