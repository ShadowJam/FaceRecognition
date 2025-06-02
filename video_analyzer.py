import cv2
import numpy as np
from collections import defaultdict
from datetime import datetime
from norfair import Tracker, Detection
from config import Config
import threading
import torch
from PIL import Image
from facenet_pytorch import MTCNN
import base64


class VideoAnalyzer:
    def __init__(self, identifier):
        self.identifier = identifier
        self.tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=Config.TRACKING_DISTANCE_THRESHOLD,
            initialization_delay=Config.TRACKING_INIT_DELAY,
            hit_counter_max=Config.TRACKING_HIT_COUNTER_MAX
        )
        self.results = {
            'recognized': {},
            'unrecognized': [],
            'analysis_info': {
                'start_time': None,
                'end_time': None,
                'video_source': None,
                'total_frames': 0,
                'processed_frames': 0
            }
        }
        self.current_frame = None
        self.frame_count = 0
        self.mtcnn = MTCNN(keep_all=True, device=identifier.device)
        self.sample_images = {}
        self.lock = threading.Lock()  # Инициализация блокировки

    def preprocess_frame(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def detect_faces(self, frame):
        frame_rgb = self.preprocess_frame(frame)
        boxes, _ = self.mtcnn.detect(frame_rgb)

        if boxes is None:
            return []

        return [{
            'box': [int(x) for x in box],
            'confidence': 0.99
        } for box in boxes]

    def process_frame(self, frame, timestamp):
        faces = self.detect_faces(frame)
        detections = []

        for face in faces:
            x1, y1, x2, y2 = face['box']
            points = np.array([
                [x1, y1],  # Top-left
                [x2, y1],  # Top-right
                [x2, y2],  # Bottom-right
                [x1, y2]  # Bottom-left
            ])

            detections.append(
                Detection(
                    points=points,
                    scores=np.array([face['confidence']] * 4)
                )
            )

        tracked_objects = self.tracker.update(detections=detections)

        with self.lock:  # Защищаем все обновления результатов
            for obj in tracked_objects:
                points = obj.estimate
                x1 = int(min(points[:, 0]))
                y1 = int(min(points[:, 1]))
                x2 = int(max(points[:, 0]))
                y2 = int(max(points[:, 1]))
                track_id = obj.id
                face_img = frame[y1:y2, x1:x2]

                if face_img.size == 0:
                    continue

                try:
                    pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    person_id, confidence = self.identifier.identify_face(pil_img)
                except Exception as e:
                    print(f"Face identification error: {e}")
                    continue

                if person_id != "Unknown":
                    if person_id not in self.results['recognized']:
                        self.results['recognized'][person_id] = {
                            'first_seen': timestamp,
                            'last_seen': timestamp,
                            'count': 1,
                            'max_confidence': confidence,
                            'sample_images': [],
                            'track_id': track_id,
                            'last_box': (x1, y1, x2, y2)
                        }
                    else:
                        self.results['recognized'][person_id].update({
                            'last_seen': timestamp,
                            'count': self.results['recognized'][person_id]['count'] + 1,
                            'max_confidence': max(
                                self.results['recognized'][person_id]['max_confidence'],
                                confidence
                            ),
                            'last_box': (x1, y1, x2, y2)
                        })

                    if len(self.results['recognized'][person_id]['sample_images']) < 10:
                        _, img_encoded = cv2.imencode('.jpg', face_img)
                        self.results['recognized'][person_id]['sample_images'].append(
                            img_encoded.tobytes()
                        )
                else:
                    self.results['unrecognized'].append({
                        'timestamp': timestamp,
                        'confidence': confidence,
                        'track_id': track_id,
                        'last_box': (x1, y1, x2, y2)
                    })
                    #if len(self.results['unrecognized']) <= 10:
                    _, img_encoded = cv2.imencode('.jpg', face_img)
                    self.results['unrecognized'][-1]['image'] = img_encoded.tobytes()

        self.frame_count += 1
        self.current_frame = self.draw_results(frame.copy())
        return faces

    def draw_results(self, frame):
        with self.lock:
            # Отрисовка распознанных лиц
            for person_id, data in self.results['recognized'].items():
                if 'last_box' in data and data['last_box'] is not None:
                    x1, y1, x2, y2 = data['last_box']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{person_id} ({data['max_confidence']:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

            # Отрисовка нераспознанных лиц
            for unknown in self.results['unrecognized']:
                if 'last_box' in unknown and unknown['last_box'] is not None:
                    x1, y1, x2, y2 = unknown['last_box']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        f"Unknown ({unknown.get('confidence', 0):.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

        return frame

    def get_results(self):
        with self.lock:
            # Конвертируем datetime в строки при возврате результатов
            results = {
                'recognized': {},
                'unrecognized': [],
                'analysis_info': {
                    'video_source': self.results['analysis_info']['video_source'],
                    'start_time': str(self.results['analysis_info']['start_time']) if self.results['analysis_info'][
                        'start_time'] else None,
                    'end_time': str(self.results['analysis_info']['end_time']) if self.results['analysis_info'][
                        'end_time'] else None,
                    'total_frames': self.results['analysis_info']['total_frames'],
                    'processed_frames': self.results['analysis_info']['processed_frames']
                }
            }

            # Обрабатываем recognized
            for person_id, data in self.results['recognized'].items():
                results['recognized'][person_id] = {
                    'first_seen': str(data['first_seen']) if data['first_seen'] else None,
                    'last_seen': str(data['last_seen']) if data['last_seen'] else None,
                    'count': data['count'],
                    'max_confidence': data['max_confidence'],
                    'sample_images': [
                        base64.b64encode(img).decode('utf-8')
                        for img in data['sample_images']
                    ]
                }

            # Обрабатываем unrecognized
            #for item in self.results['unrecognized']:
            #    results['unrecognized'].append({
            #        'timestamp': str(item['timestamp']) if item['timestamp'] else None,
            #        'confidence': item['confidence'],
            #        'image': base64.b64encode(item['image']).decode('utf-8') if 'image' in item else None
            #    })

            filtered_unrecognized = [
                item for item in self.results['unrecognized']
                if item.get('image') and item.get('confidence', 0) > 0.3
            ]
            filtered_unrecognized.sort(key=lambda x: x['confidence'], reverse=True)
            filtered_unrecognized = filtered_unrecognized[:10]

            results['unrecognized'] = [
                {
                    'timestamp': str(item['timestamp']) if item['timestamp'] else None,
                    'confidence': item['confidence'],
                    'image': base64.b64encode(item['image']).decode('utf-8') if 'image' in item else None
                }
                for item in filtered_unrecognized
            ]

            return results