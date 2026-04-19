import cv2
import mediapipe as mp
import csv
import os
import time
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

DATA_FILE = "sign_sequence_data.csv"
FRAMES_PER_SAMPLE = 30
FEATURES_PER_FRAME = 126
FRAME_SKIP = 2  # Process every 2nd frame for faster collection

def normalize_landmarks(landmarks):
    """Normalize landmarks to be wrist-relative and scale-invariant"""
    if not landmarks:
        return None
    
    # Extract all points
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # Make wrist-relative (wrist is landmark 0)
    wrist = points[0]
    points = points - wrist
    
    # Scale normalization (based on hand size)
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
                features.extend([0] * 63)  # 21 landmarks × 3 coords
    else:
        features.extend([0] * 126)  # Both hands empty
    return features

def augment_sequence(sequence):
    """Generate augmented versions of the sequence"""
    augmented = [sequence]  # Original
    
    # Time stretching (speed variation)
    if len(sequence) >= 20:
        # Faster version (skip every 3rd frame, then duplicate to match length)
        faster = sequence[::3]
        while len(faster) < FRAMES_PER_SAMPLE:
            faster.extend(faster[:FRAMES_PER_SAMPLE - len(faster)])
        augmented.append(faster[:FRAMES_PER_SAMPLE])
        
        # Slower version (duplicate frames)
        slower = []
        for frame in sequence:
            slower.extend([frame, frame])
        augmented.append(slower[:FRAMES_PER_SAMPLE])
    
    # Small noise addition (simulate tracking jitter)
    noisy = []
    for frame in sequence:
        noisy_frame = [f + np.random.normal(0, 0.01) for f in frame]
        noisy.append(noisy_frame)
    augmented.append(noisy)
    
    return augmented

def record_sequence(cap, hands, label, moving=False, sample_num=1, total_samples=1):
    sequence = []
    frame_count = 0
    
    print(f"Recording sample {sample_num}/{total_samples} for '{label}' [{'MOVING' if moving else 'STATIC'}]...")
    print("Get ready! Recording starts in 2 seconds...")
    time.sleep(2)

    while len(sequence) < FRAMES_PER_SAMPLE:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        
        # Skip frames for faster processing
        if frame_count % FRAME_SKIP != 0:
            continue
            
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(img_rgb)

        features = extract_features(res.multi_hand_landmarks)
        sequence.append(features)

        # Draw landmarks
        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Overlay progress with color coding
        progress_pct = len(sequence) / FRAMES_PER_SAMPLE
        color = (0, int(255 * progress_pct), int(255 * (1 - progress_pct)))
        
        cv2.rectangle(frame, (0,0), (500,80), (0,0,0), -1)
        cv2.putText(frame, f"Recording: {len(sequence)}/{FRAMES_PER_SAMPLE}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Label: {label}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        cv2.imshow("Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Recording interrupted!")
            break

    return sequence

def main():
    print("=== Enhanced Data Collection with Augmentation ===")
    print("Press 'q' in the camera window to quit at any time.\n")

    file_exists = os.path.isfile(DATA_FILE)
    with open(DATA_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = [f"f{i}" for i in range(FEATURES_PER_FRAME * FRAMES_PER_SAMPLE)] + ["label"]
            writer.writerow(header)

        while True:
            # Ask for gesture label
            label = input("\nEnter gesture label (or 'q' to quit): ").strip()
            if label.lower() == "q":
                print("Exiting data collection.")
                break
            if not label:
                print("⚠️ Please enter a valid label.")
                continue

            # Ask if gesture is moving
            moving_input = input("Is this gesture moving? (y/n): ").strip().lower()
            moving = moving_input == "y"

            # Ask number of samples
            try:
                total_samples = int(input("How many samples to record? (recommend 5-10): ").strip())
                if total_samples < 1:
                    print("⚠️ Must record at least 1 sample.")
                    continue
            except ValueError:
                print("⚠️ Please enter a valid integer.")
                continue

            # Ask about augmentation
            aug_input = input("Apply data augmentation? (y/n, recommended): ").strip().lower()
            use_augmentation = aug_input == "y"

            # Open webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Cannot open webcam. Exiting.")
                break

            samples_saved = 0
            with mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
                for sample_num in range(1, total_samples + 1):
                    sequence = record_sequence(cap, hands, label, moving, sample_num, total_samples)
                    
                    if sequence and len(sequence) == FRAMES_PER_SAMPLE:
                        # Save original
                        flat_sequence = [f for frame in sequence for f in frame]
                        flat_sequence.append(label)
                        writer.writerow(flat_sequence)
                        samples_saved += 1
                        
                        # Save augmented versions
                        if use_augmentation:
                            augmented_sequences = augment_sequence(sequence)
                            for aug_seq in augmented_sequences[1:]:  # Skip original
                                flat_aug = [f for frame in aug_seq for f in frame]
                                flat_aug.append(label)
                                writer.writerow(flat_aug)
                                samples_saved += 1
                        
                        print(f"✅ Saved sample {sample_num}/{total_samples} for '{label}' "
                              f"({'with augmentation' if use_augmentation else 'no augmentation'})")
                    else:
                        print(f"⚠️ Sample {sample_num} incomplete, not saved.")
                    
                    # Small pause between samples
                    if sample_num < total_samples:
                        time.sleep(1)

            cap.release()
            cv2.destroyAllWindows()
            print(f"\n📊 Total samples saved for '{label}': {samples_saved}")

if __name__ == "__main__":
    main()