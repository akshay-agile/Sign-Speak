import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import time
from collections import deque
import threading

MODEL_FILE = "sign_lstm_model.keras"
ENCODER_FILE = "label_encoder.joblib"

FRAMES_PER_SAMPLE = 30
FEATURES_PER_FRAME = 126
FRAME_SKIP = 2  # Process every 2nd frame for speed
PREDICTION_INTERVAL = 15  # Predict every N frames
CONFIDENCE_THRESHOLD = 0.75  # Higher threshold for better accuracy
SMOOTHING_WINDOW = 3  # Smooth predictions over last N predictions

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def normalize_landmarks(landmarks):
    """Normalize landmarks (same as in collect_data.py)"""
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
    """Extract normalized features from both hands"""
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

class TTSThread:
    """Non-blocking Text-to-Speech"""
    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()
        self.running = True
        
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)  # Faster speech
            self.enabled = True
            
            # Start TTS thread
            self.thread = threading.Thread(target=self._speak_worker, daemon=True)
            self.thread.start()
        except:
            print("⚠️ TTS unavailable")
            self.enabled = False
    
    def _speak_worker(self):
        """Background worker for TTS"""
        while self.running:
            with self.lock:
                if self.queue:
                    text = self.queue.pop(0)
                    try:
                        self.engine.say(text)
                        self.engine.runAndWait()
                    except:
                        pass
            time.sleep(0.1)
    
    def speak(self, text):
        """Add text to speech queue"""
        if self.enabled:
            with self.lock:
                if text not in self.queue:  # Avoid duplicates
                    self.queue.append(text)

def smooth_predictions(prediction_history, current_pred):
    """Smooth predictions using majority voting"""
    prediction_history.append(current_pred)
    if len(prediction_history) >= SMOOTHING_WINDOW:
        # Return most common prediction in window
        return max(set(prediction_history), key=prediction_history.count)
    return current_pred

def main():
    print("=== Real-time Sign Recognition (Optimized) ===")
    
    # Load model
    print("📥 Loading model...")
    model = tf.keras.models.load_model(MODEL_FILE)
    le = joblib.load(ENCODER_FILE)
    print(f"✅ Model loaded. Recognizing {len(le.classes_)} signs: {le.classes_}")

    # Initialize TTS
    tts = TTSThread()

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    # Set camera properties for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # State variables
    sequence = deque(maxlen=FRAMES_PER_SAMPLE)
    predicted_label = "Waiting..."
    confidence = 0.0
    last_spoken = ""
    speak_cooldown = 2.0
    last_speak_time = 0
    frame_count = 0
    prediction_history = deque(maxlen=SMOOTHING_WINDOW)
    
    # FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0

    print("\n🎥 Starting camera... Press 'q' to quit")
    print("💡 Tip: Hold gesture steady for best results\n")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            fps_frame_count += 1
            
            # Calculate FPS every second
            if time.time() - fps_start_time > 1.0:
                fps = fps_frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0

            frame = cv2.flip(frame, 1)
            
            # Process only every Nth frame
            if frame_count % FRAME_SKIP == 0:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(img_rgb)

                # Extract features
                features = extract_features(res.multi_hand_landmarks)
                sequence.append(features)

                # Draw landmarks
                if res.multi_hand_landmarks:
                    for hand_landmarks in res.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(255,255,255), thickness=2)
                        )

                # Predict every N frames when sequence is full
                if len(sequence) == FRAMES_PER_SAMPLE and frame_count % PREDICTION_INTERVAL == 0:
                    X = np.array(sequence).reshape(1, FRAMES_PER_SAMPLE, FEATURES_PER_FRAME)
                    pred_probs = model.predict(X, verbose=0)[0]
                    pred_index = np.argmax(pred_probs)
                    confidence = pred_probs[pred_index]
                    
                    # Only accept predictions above threshold
                    if confidence >= CONFIDENCE_THRESHOLD and res.multi_hand_landmarks:
                        label = le.inverse_transform([pred_index])[0]
                        
                        # Smooth predictions
                        smoothed_label = smooth_predictions(prediction_history, label)
                        predicted_label = smoothed_label
                        
                        # Speak if label changed and cooldown passed
                        current_time = time.time()
                        if (smoothed_label != last_spoken and 
                            current_time - last_speak_time > speak_cooldown):
                            tts.speak(smoothed_label)
                            last_spoken = smoothed_label
                            last_speak_time = current_time
                    elif not res.multi_hand_landmarks:
                        predicted_label = "No hand"
                        confidence = 0.0
                        prediction_history.clear()
                    else:
                        predicted_label = "Low confidence"

            # UI Overlay
            overlay_height = 120
            cv2.rectangle(frame, (0, 0), (frame.shape[1], overlay_height), (0, 0, 0), -1)
            
            # Prediction with confidence bar
            label_text = f"Sign: {predicted_label}"
            conf_text = f"Confidence: {confidence*100:.1f}%"
            fps_text = f"FPS: {fps:.1f}"
            buffer_text = f"Buffer: {len(sequence)}/{FRAMES_PER_SAMPLE}"
            
            cv2.putText(frame, label_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, conf_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, fps_text, (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, buffer_text, (10, 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Confidence bar
            if confidence > 0:
                bar_width = int(300 * confidence)
                bar_color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
                cv2.rectangle(frame, (frame.shape[1] - 320, 20), 
                            (frame.shape[1] - 320 + bar_width, 40), bar_color, -1)
                cv2.rectangle(frame, (frame.shape[1] - 320, 20), 
                            (frame.shape[1] - 20, 40), (100, 100, 100), 2)

            cv2.imshow("Sign Interpreter (Optimized)", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):  # Reset buffer
                sequence.clear()
                prediction_history.clear()
                print("🔄 Buffer reset")

    cap.release()
    cv2.destroyAllWindows()
    tts.running = False
    print("\n✅ Stopped")

if __name__ == "__main__":
    main()