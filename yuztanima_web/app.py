from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import face_recognition
import pickle
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Verileri yükle
try:
    with open('face_encodings.pkl', 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
except FileNotFoundError:
    known_face_encodings = []
    known_face_names = []

# Kamerayı başlat
video_capture = None

def start_camera():
    global video_capture
    if video_capture is None or not video_capture.isOpened():
        video_capture = cv2.VideoCapture(0)
        return "Kamera başarıyla açıldı!"
    return "Kamera zaten açık!"

def stop_camera():
    global video_capture
    if video_capture is not None and video_capture.isOpened():
        video_capture.release()
        return "Kamera başarıyla kapatıldı!"
    return "Kamera zaten kapalı!"

@socketio.on('start_camera')
def handle_start_camera():
    status = start_camera()
    emit('status', {'status': status})

@socketio.on('stop_camera')
def handle_stop_camera():
    status = stop_camera()
    emit('status', {'status': status})

@socketio.on('stream')
def handle_stream():
    global video_capture
    if video_capture is None or not video_capture.isOpened():
        emit('status', {'error': 'Kamera kapalı!'})
        return
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            emit('status', {'error': 'Kamera verisi alınamıyor!'})
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        name = "Bilinmiyor"  # Varsayılan değer

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        emit('video_frame', frame_bytes)
        socketio.sleep(0.1)  # Bu satır, aşırı CPU kullanımını önlemek için eklendi

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    name = request.form['name']

    # Resmi yükle ve yüzleri bul
    image = face_recognition.load_image_file(file)
    face_encodings = face_recognition.face_encodings(image)

    if len(face_encodings) > 0:
        new_face_encoding = face_encodings[0]
        known_face_encodings.append(new_face_encoding)
        known_face_names.append(name)

        # Verileri kaydet
        with open('face_encodings.pkl', 'wb') as f:
            pickle.dump((known_face_encodings, known_face_names), f)
        
        return jsonify({'identity': name})
    else:
        return jsonify({'error': 'Yüz bulunamadı!'}), 400

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)