from flask import Flask, flash, redirect, render_template, request, Response, jsonify, url_for, send_from_directory
from flask_login import LoginManager, login_required, current_user
import os
import threading
import cv2
import time
from datetime import datetime
from werkzeug.utils import secure_filename
import torch
import shutil
from pathlib import Path
from base64 import b64encode

from config import Config
from face_identifier import FaceIdentifier
from video_analyzer import VideoAnalyzer
from auth import login_manager, auth_bp


#app = Flask(__name__)
#app.jinja_env.filters['datetimeformat'] = datetimeformat

app = Flask(__name__)
app.secret_key = 'super-secret-key'
app.config.from_object(Config)
Config.init_app(app)
login_manager.init_app(app)
app.register_blueprint(auth_bp)

# Инициализация компонентов (один раз при запуске)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
identifier = FaceIdentifier(device=device)
analyzer = VideoAnalyzer(identifier)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def get_video_files():
    """Получаем список видеофайлов из uploads/video"""
    if not Config.UPLOAD_FOLDER.exists():
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        return []
    return sorted(Config.UPLOAD_FOLDER.glob('*'))

# Добавляем в конфигурацию
app.config['UPLOAD_PHOTO_FOLDER'] = 'uploads/photo'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB




@app.route('/')
@login_required
def index():
    has_data = len(identifier.known_faces) > 0
    video_files = get_video_files()
    return render_template('index.html',
                         has_data=has_data,
                         video_files=video_files,
                         known_faces_count=len(identifier.known_faces))


@app.route('/register', methods=['GET', 'POST'])
@login_required
def register():
    if request.method == 'POST':
        person_id = request.form['person_id'].strip()
        if not person_id:
            flash('Please enter person ID', 'error')
            return redirect(request.url)

        # Проверяем, есть ли уже такой пользователь
        existing_dir = Config.KNOWN_FACES_FOLDER / person_id
        if existing_dir.exists():
            flash(f'Person {person_id} already exists!', 'error')
            return redirect(request.url)

        saved_paths = []
        for file in request.files.getlist('photos'):
            if file and allowed_file(file.filename):
                # Создаем директорию только если есть валидные файлы
                if not existing_dir.exists():
                    os.makedirs(existing_dir)

                filename = secure_filename(file.filename)
                filepath = existing_dir / filename
                file.save(filepath)
                saved_paths.append(str(filepath))

        if saved_paths:
            try:
                success = identifier.add_person(person_id, saved_paths)
                if success:
                    flash(f'Successfully registered {person_id} with {len(saved_paths)} images', 'success')
                    return redirect(url_for('index'))
                else:
                    # Удаляем сохраненные файлы если не удалось добавить
                    shutil.rmtree(existing_dir)
                    flash('Failed to process face images', 'error')
            except Exception as e:
                if existing_dir.exists():
                    shutil.rmtree(existing_dir)
                flash(f'Error: {str(e)}', 'error')
        else:
            flash('No valid images uploaded', 'error')

        return redirect(request.url)

    return render_template('register.html')


@app.route('/upload_video', methods=['POST'])
@login_required
def upload_video():
    if 'video' not in request.files:
        flash('No video file uploaded', 'error')
        return redirect(url_for('index'))

    video_file = request.files['video']
    if video_file.filename == '':
        flash('No selected video file', 'error')
        return redirect(url_for('index'))

    if video_file and allowed_file(video_file.filename):
        filename = secure_filename(video_file.filename)
        filepath = Config.UPLOAD_FOLDER / filename
        video_file.save(filepath)
        flash(f'Video {filename} uploaded successfully', 'success')

    return redirect(url_for('index'))


@app.route('/video_feed')
@login_required
def video_feed():
    if not identifier.known_faces:
        return Response(
            open(Config.BASE_DIR / 'static' / 'no_data.jpg', 'rb').read(),
            mimetype='image/jpeg'
        )

    def generate():
        while True:
            if analyzer.current_frame is not None:
                ret, jpeg = cv2.imencode('.jpg', analyzer.current_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.033)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_analysis', methods=['POST'])
@login_required
def start_analysis():
    if not identifier.known_faces:
        flash('No registered faces found', 'error')
        return redirect(url_for('index'))

    video_source = request.form.get('video_source', '')
    if not video_source:
        flash('No video source provided', 'error')
        return redirect(url_for('video'))

    # Инициализация анализа
    analyzer.results = {
        'recognized': {},
        'unrecognized': [],
        'analysis_info': {
            'start_time': str(datetime.now()),
            'end_time': None,
            'video_source': video_source,
            'total_frames': 0,
            'processed_frames': 0
        }
    }
    analyzer.frame_count = 0

    def analyze_video():
        cap = cv2.VideoCapture(str(Config.UPLOAD_FOLDER / video_source))
        analyzer.results['analysis_info']['total_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            analyzer.process_frame(frame, timestamp)
            analyzer.results['analysis_info']['processed_frames'] += 1

        cap.release()
        analyzer.results['analysis_info']['end_time'] = str(datetime.now())

    threading.Thread(target=analyze_video, daemon=True).start()
    flash(f'Analysis started for {video_source}', 'info')
    return redirect(url_for('video'))


@app.route('/get_results')
@login_required
def get_results():
    return jsonify(analyzer.get_results())


@app.route('/reports')
@login_required
def reports():
    results = analyzer.get_results()
    if not results['recognized'] and not results['unrecognized']:
        flash('No analysis data available', 'error')
        return redirect(url_for('index'))

    return render_template('reports.html', report=results)


@app.route('/video')
@login_required
def video():
    if not identifier.known_faces:
        flash('Please register people before video analysis', 'error')
        return redirect(url_for('index'))

    video_files = get_video_files()
    return render_template('video.html', video_files=video_files)


@app.route('/clear_data', methods=['POST'])
@login_required
def clear_data():
    try:
        # Удаляем файл с сохраненными лицами
        faces_file = Config.KNOWN_FACES_FOLDER / 'faces.dat'
        if faces_file.exists():
            os.remove(faces_file)

        # Пересоздаем идентификатор для очистки памяти
        global identifier
        identifier = FaceIdentifier(device=device)
        analyzer.face_identifier = identifier

        flash('Face recognition data cleared successfully', 'success')
    except Exception as e:
        flash(f'Error clearing data: {str(e)}', 'error')
    return redirect(url_for('index'))


@app.route('/view_people')
@login_required
def view_people():
    people = []
    for person_id in os.listdir(Config.KNOWN_FACES_FOLDER):
        person_dir = Config.KNOWN_FACES_FOLDER / person_id
        if person_dir.is_dir():
            images = list(person_dir.glob('*.jpg')) + list(person_dir.glob('*.png'))
            if images:  # Показываем только пользователей с изображениями
                people.append({
                    'id': person_id,
                    'images': images,
                    'count': len(images)
                })
    return render_template('view_people.html', people=sorted(people, key=lambda x: x['id']))


@app.route('/delete_person/<person_id>', methods=['POST'])
@login_required
def delete_person(person_id):
    try:
        person_dir = Config.KNOWN_FACES_FOLDER / person_id
        if person_dir.exists():
            shutil.rmtree(person_dir)
            if person_id in identifier.known_faces:
                del identifier.known_faces[person_id]
                identifier.save_known_faces()
            flash(f'Person {person_id} deleted successfully', 'success')
    except Exception as e:
        flash(f'Error deleting person: {str(e)}', 'error')
    return redirect(url_for('view_people'))

@app.route('/delete_image/<person_id>/<image_name>', methods=['POST'])
@login_required
def delete_image(person_id, image_name):
    try:
        image_path = Config.KNOWN_FACES_FOLDER / person_id / image_name
        if image_path.exists():
            image_path.unlink()
            # Перезагружаем данные пользователя
            if person_id in identifier.known_faces:
                person_dir = Config.KNOWN_FACES_FOLDER / person_id
                image_paths = [str(p) for p in person_dir.glob('*') if p.is_file()]
                if image_paths:
                    identifier.add_person(person_id, image_paths)
                else:
                    del identifier.known_faces[person_id]
                    identifier.save_known_faces()
            flash('Image deleted successfully', 'success')
    except Exception as e:
        flash(f'Error deleting image: {str(e)}', 'error')
    return redirect(url_for('view_people'))


@app.route('/uploads/photo/<path:filename>')
@login_required
def uploaded_photo(filename):
    return send_from_directory(Config.KNOWN_FACES_FOLDER, filename)


@app.template_filter('b64encode')
def base64_encode_filter(data):
    if isinstance(data, str):
        return b64encode(data.encode()).decode('utf-8')
    return b64encode(data).decode('utf-8')

if __name__ == '__main__':
    # Инициализация данных
    identifier.load_known_faces()
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
