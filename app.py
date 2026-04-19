import os
from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime
import json
from collections import deque
import threading
import time
import pyttsx3
import re

app = Flask(__name__)

# Configuration
MODEL_FILE = "sign_lstm_model.keras"
ENCODER_FILE = "label_encoder.joblib"
HISTORY_FILE = "translation_history.json"
UPLOAD_FOLDER = "uploads"
SIGN_INDEX_FILE = "sign_index.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
model = None
label_encoder = None
camera = None
camera_active = False
camera_lock = threading.Lock()
tts_lock = threading.Lock()
sign_index = {} 

# Live prediction state
current_prediction = {
    'sign': 'Waiting...',
    'confidence': 0.0,
    'timestamp': time.time()
}
prediction_lock = threading.Lock()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

FRAMES_PER_SAMPLE = 30
FEATURES_PER_FRAME = 126
CONFIDENCE_THRESHOLD = 0.75

# Initialize TTS - FIXED VERSION
def speak_text_sync(text):
    """Synchronous TTS function that creates and destroys engine each time"""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        del engine
        return True
    except Exception as e:
        print(f"⚠️ TTS error: {e}")
        return False

# Load sign index
def load_sign_index():
    global sign_index
    try:
        if os.path.exists(SIGN_INDEX_FILE):
            with open(SIGN_INDEX_FILE, 'r') as f:
                sign_index = json.load(f)
            print(f"✅ Loaded {len(sign_index)} signs from index")
        else:
            print("⚠️ sign_index.json not found. Run index_dataset.py first!")
    except Exception as e:
        print(f"❌ Error loading sign index: {e}")

# Load model at startup
def load_model():
    global model, label_encoder
    try:
        if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
            model = tf.keras.models.load_model(MODEL_FILE)
            label_encoder = joblib.load(ENCODER_FILE)
            print("✅ Model loaded successfully")
            return True
        else:
            print("⚠️ Model files not found")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

model_loaded = load_model()
load_sign_index()

def normalize_landmarks(landmarks):
    if not landmarks:
        return None
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = points[0]
    points = points - wrist
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist
    return points.flatten()

def extract_features(multi_hand_landmarks):
    features = []
    if multi_hand_landmarks:
        for i in range(2):
            if i < len(multi_hand_landmarks):
                normalized = normalize_landmarks(multi_hand_landmarks[i].landmark)
                features.extend(normalized)
            else:
                features.extend([0] * 63)
    else:
        features.extend([0] * 126)
    return np.array(features)

def update_prediction(sign, confidence):
    global current_prediction
    if confidence >= CONFIDENCE_THRESHOLD:
        with prediction_lock:
            current_prediction = {
                'sign': sign,
                'confidence': float(confidence),
                'timestamp': time.time()
            }

def generate_frames():
    global camera, camera_active
    sequence = deque(maxlen=FRAMES_PER_SAMPLE)
    frame_count = 0
    prediction_interval = 10
    last_predicted_sign = None
    last_predicted_confidence = 0.0
    hand_detection_buffer = deque(maxlen=5)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        while camera_active:
            with camera_lock:
                if camera is None:
                    break
                success, frame = camera.read()
                if not success:
                    continue

            frame_count += 1
            frame = cv2.flip(frame, 1)
            current_hand_detected = False

            if frame_count % 2 == 0:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                current_hand_detected = results.multi_hand_landmarks is not None
                hand_detection_buffer.append(current_hand_detected)
                features = extract_features(results.multi_hand_landmarks)
                sequence.append(features)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(255,255,255), thickness=2)
                        )

            hand_detected_smooth = sum(hand_detection_buffer) >= 2
            display_label = "No hand detected"
            display_confidence = 0.0

            if hand_detected_smooth:
                if len(sequence) == FRAMES_PER_SAMPLE and model is not None and frame_count % prediction_interval == 0:
                    X = np.array(sequence).reshape(1, FRAMES_PER_SAMPLE, FEATURES_PER_FRAME)
                    pred_probs = model.predict(X, verbose=0)[0]
                    pred_index = np.argmax(pred_probs)
                    confidence = pred_probs[pred_index]

                    if confidence >= CONFIDENCE_THRESHOLD:
                        predicted_label = label_encoder.inverse_transform([pred_index])[0]
                        display_label = predicted_label
                        display_confidence = confidence

                        if predicted_label != last_predicted_sign:
                            update_prediction(predicted_label, confidence)
                            last_predicted_sign = predicted_label
                            last_predicted_confidence = confidence
                    else:
                        display_label = "Detecting..."
                        display_confidence = confidence
                elif last_predicted_sign:
                    display_label = last_predicted_sign
                    display_confidence = last_predicted_confidence
                else:
                    display_label = "Detecting..."
            else:
                display_label = "No hand detected"
                display_confidence = 0.0

            # Add overlay
            overlay_height = 100
            cv2.rectangle(frame, (0, 0), (frame.shape[1], overlay_height), (0, 0, 0), -1)

            if hand_detected_smooth and display_confidence >= CONFIDENCE_THRESHOLD:
                text_color = (0, 255, 0)
            elif hand_detected_smooth:
                text_color = (0, 165, 255)
            else:
                text_color = (100, 100, 100)

            cv2.putText(frame, f"Sign: {display_label}", (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
            cv2.putText(frame, f"Confidence: {display_confidence*100:.1f}%", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            buffer_text = f"Buffer: {len(sequence)}/{FRAMES_PER_SAMPLE}"
            cv2.putText(frame, buffer_text, (frame.shape[1] - 200, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.circle(frame, (frame.shape[1] - 30, 70), 8, (0, 0, 255), -1)
            cv2.putText(frame, "LIVE", (frame.shape[1] - 75, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Helper function
def normalize_for_search(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    text = text.strip('_')
    return text

# Routes
@app.route('/')
def index():
    signs = label_encoder.classes_.tolist() if label_encoder else []
    return render_template('index.html', signs=signs)

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model_loaded,
        'camera_active': camera_active,
        'signs_count': len(label_encoder.classes_) if label_encoder else 0,
        'sign_index_count': len(sign_index)
    })

@app.route('/current_prediction')
def get_current_prediction():
    with prediction_lock:
        return jsonify(current_prediction)

@app.route('/clear_prediction', methods=['POST'])
def clear_prediction():
    global current_prediction
    with prediction_lock:
        current_prediction = {
            'sign': 'Waiting...',
            'confidence': 0.0,
            'timestamp': time.time()
        }
    return jsonify({'success': True})

@app.route('/speak', methods=['POST'])
def speak_text():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'success': False, 'error': 'No text provided'})
    
    print(f"🔊 Speaking: {text}")  # Debug log
    
    try:
        # Use threading with lock to prevent concurrent TTS calls
        def speak_thread():
            with tts_lock:
                speak_text_sync(text)
        
        thread = threading.Thread(target=speak_thread)
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/text_to_sign', methods=['POST'])
def text_to_sign():
    data = request.get_json()
    text_input = data.get('text', '').strip()

    if not text_input:
        return jsonify({'success': False, 'error': 'No text provided'})

    words = text_input.split()
    results = []
    i = 0

    while i < len(words):
        matched = False
        max_len = min(5, len(words) - i)

        for length in range(max_len, 0, -1):
            phrase = ' '.join(words[i:i+length])
            normalized_phrase = normalize_for_search(phrase)

            if normalized_phrase in sign_index:
                sign_data = sign_index[normalized_phrase]
                result = {
                    'word': phrase.upper(),
                    'found': True,
                    'type': sign_data['type'],
                    'original_name': sign_data['original_name']
                }

                if sign_data['type'] == 'video':
                    result['video'] = sign_data['video_path']
                else:
                    result['image'] = sign_data['image_path']

                results.append(result)
                i += length
                matched = True
                break

        if not matched:
            results.append({
                'word': words[i].upper(),
                'found': False,
                'message': f'Sign not available for "{words[i]}"'
            })
            i += 1

    return jsonify({
        'success': True,
        'results': results,
        'total_words': len(words),
        'found_count': sum(1 for r in results if r['found'])
    })

@app.route('/available_signs')
def available_signs():
    return jsonify({
        'signs': sorted(list(sign_index.keys())),
        'count': len(sign_index)
    })

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera, camera_active, current_prediction

    if not model_loaded:
        return jsonify({'success': False, 'error': 'Model not loaded'})

    with camera_lock:
        if camera is not None:
            return jsonify({'success': False, 'error': 'Camera already running'})

        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            camera = None
            return jsonify({'success': False, 'error': 'Could not open camera'})

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera_active = True

        with prediction_lock:
            current_prediction = {
                'sign': 'Waiting...',
                'confidence': 0.0,
                'timestamp': time.time()
            }

        print("📹 Camera started")
        return jsonify({'success': True})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera, camera_active

    with camera_lock:
        camera_active = False
        if camera is not None:
            camera.release()
            camera = None

    print("📹 Camera stopped")
    return jsonify({'success': True})

@app.route('/video_feed')
def video_feed():
    if not camera_active:
        return "Camera not started", 400
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_video():
    if not model_loaded:
        return jsonify({'success': False, 'error': 'Model not loaded'})

    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file'})

    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        cap = cv2.VideoCapture(filepath)
        sequence = deque(maxlen=FRAMES_PER_SAMPLE)
        detected_signs = []
        frame_count = 0

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        ) as hands:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % 2 != 0:
                    continue

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                features = extract_features(results.multi_hand_landmarks)
                sequence.append(features)

                if len(sequence) == FRAMES_PER_SAMPLE:
                    X = np.array(sequence).reshape(1, FRAMES_PER_SAMPLE, FEATURES_PER_FRAME)
                    pred_probs = model.predict(X, verbose=0)[0]
                    pred_index = np.argmax(pred_probs)
                    confidence = pred_probs[pred_index]

                    if confidence >= CONFIDENCE_THRESHOLD and results.multi_hand_landmarks:
                        sign = label_encoder.inverse_transform([pred_index])[0]
                        detected_signs.append({
                            'sign': sign,
                            'confidence': float(confidence),
                            'frame': frame_count
                        })

        cap.release()
        os.remove(filepath)

        translation = ' '.join([s['sign'] for s in detected_signs])

        return jsonify({
            'success': True,
            'translation': translation,
            'detected_signs': detected_signs,
            'total_frames': frame_count
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    if not model_loaded:
        print("\n⚠️ WARNING: Model not loaded!")
        print("Please run: python train_model.py")
    else:
        print(f"\n✅ Ready! Recognizing {len(label_encoder.classes_)} signs")

    if not sign_index:
        print("\n⚠️ WARNING: No signs indexed!")
        print("Please run: python index_dataset.py")
    else:
        print(f"✅ {len(sign_index)} signs available for text-to-sign")

    print("\n🌐 Open browser to: http://localhost:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)