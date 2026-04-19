import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import joblib
import matplotlib.pyplot as plt

DATA_FILE = "sign_sequence_data.csv"
MODEL_FILE = "sign_lstm_model.keras"
ENCODER_FILE = "label_encoder.joblib"

FRAMES_PER_SAMPLE = 30
FEATURES_PER_FRAME = 126

def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("📊 Training history saved to training_history.png")

def build_improved_model(input_shape, num_classes):
    """Build an optimized LSTM model with bidirectional layers"""
    model = Sequential([
        # Bidirectional LSTM captures temporal patterns in both directions
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.4),
        
        # Second LSTM layer
        Bidirectional(LSTM(32)),
        BatchNormalization(),
        Dropout(0.4),
        
        # Dense layers
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation="relu"),
        Dropout(0.2),
        
        # Output layer
        Dense(num_classes, activation="softmax")
    ])
    
    # Use a lower learning rate for better convergence
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def main():
    print("📥 Loading data...")
    df = pd.read_csv(DATA_FILE)
    if df.empty:
        raise ValueError("❌ Dataset is empty!")

    # Ensure all labels are strings
    df["label"] = df["label"].astype(str)
    
    print(f"📊 Dataset info:")
    print(f"   Total samples: {len(df)}")
    print(f"   Unique labels: {df['label'].nunique()}")
    print(f"   Labels: {df['label'].unique().tolist()}")
    print(f"\n   Samples per label:")
    print(df['label'].value_counts())

    X = df.drop("label", axis=1).values
    y = df["label"].values

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    # Reshape X
    X = X.reshape(-1, FRAMES_PER_SAMPLE, FEATURES_PER_FRAME)
    print(f"\n✅ Data shape: {X.shape} (samples, frames, features)")

    # Check if we have enough samples
    min_samples_needed = max(10, len(np.unique(y_encoded)) * 2)
    if len(X) < min_samples_needed:
        print(f"⚠️ Warning: Only {len(X)} samples. Recommend at least {min_samples_needed} for good training.")
        print("   Consider collecting more data or using augmentation.")

    # Split data
    test_size = min(0.2, max(0.1, 1.0 / len(X)))  # Adaptive test size
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=test_size, random_state=42, stratify=y_encoded
    )
    
    print(f"\n🔀 Train/Test split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")

    print("\n⚙️ Building optimized LSTM model...")
    model = build_improved_model(
        input_shape=(FRAMES_PER_SAMPLE, FEATURES_PER_FRAME),
        num_classes=len(le.classes_)
    )
    
    print(model.summary())

    # Callbacks for better training
    callbacks = [
        # Stop if validation loss doesn't improve
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when stuck
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # Save best model during training
        ModelCheckpoint(
            "best_model_checkpoint.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0
        )
    ]

    print("\n🚀 Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,  # More epochs with early stopping
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\n📈 Evaluating model...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Test Accuracy: {acc*100:.2f}%")
    print(f"✅ Test Loss: {loss:.4f}")

    # Per-class accuracy
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    print("\n📊 Per-class accuracy:")
    for i, label in enumerate(le.classes_):
        mask = y_test_classes == i
        if mask.sum() > 0:
            class_acc = (y_pred_classes[mask] == i).mean()
            print(f"   {label}: {class_acc*100:.1f}% ({mask.sum()} samples)")

    # Save model and encoder
    model.save(MODEL_FILE)
    joblib.dump(le, ENCODER_FILE)
    print(f"\n💾 Model saved to {MODEL_FILE}")
    print(f"💾 Label encoder saved to {ENCODER_FILE}")
    
    # Plot training history
    try:
        plot_training_history(history)
    except Exception as e:
        print(f"⚠️ Could not plot training history: {e}")

    # Recommendations
    print("\n💡 Recommendations:")
    if acc < 0.8:
        print("   - Accuracy is low. Collect more training samples (aim for 20+ per gesture)")
        print("   - Use data augmentation during collection")
        print("   - Ensure consistent lighting and background")
    if acc > 0.95 and len(X_train) < 50:
        print("   - High accuracy with few samples may indicate overfitting")
        print("   - Test with new, unseen data")

if __name__ == "__main__":
    main()