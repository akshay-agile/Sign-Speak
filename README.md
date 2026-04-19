# Sign Language Web App

A Flask-based web application for Indian Sign Language (ISL) recognition and translation.

The app supports:
- Real-time sign recognition from webcam
- Video upload for sign prediction
- Text-to-sign lookup using indexed image/video assets
- Speech output using text-to-speech

## Features

- Live sign detection using MediaPipe hand landmarks
- Sequence classification using a TensorFlow LSTM model
- Confidence-based prediction filtering
- Text-to-sign mapping from indexed static assets
- API endpoints for health checks and live prediction state

## Tech Stack

- Python 3.10+
- Flask
- OpenCV
- MediaPipe
- TensorFlow/Keras
- scikit-learn + joblib

## Project Structure

- [app.py](app.py): Main Flask app and API routes
- [train_model.py](train_model.py): Model training script
- [index_dataset.py](index_dataset.py): Builds sign index JSON
- [requirements.txt](requirements.txt): Python dependencies
- [templates/index.html](templates/index.html): Main UI template
- [static/sign_images/](static/sign_images/): Sign images used by text-to-sign
- [sign_lstm_model.keras](sign_lstm_model.keras): Trained model
- [label_encoder.joblib](label_encoder.joblib): Label encoder
- [sign_index.json](sign_index.json): Indexed sign lookup metadata

## Setup

### 1. Clone the repository

```powershell
git clone <your-repo-url>
cd sign-language-webapp
```

### 2. Create and activate virtual environment

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyttsx3
```

## Run the App

```powershell
py app.py
```

Open in browser:
- http://localhost:5000

## First-Run Notes

If startup logs show model/index warnings:
- Run `py train_model.py` to generate/update model artifacts.
- Run `py index_dataset.py` to generate/update [sign_index.json](sign_index.json).

## API Endpoints (Quick Reference)

- `GET /` : Main web interface
- `GET /health` : Service and model status
- `POST /start_camera` : Start webcam stream
- `POST /stop_camera` : Stop webcam stream
- `GET /video_feed` : MJPEG live feed
- `GET /current_prediction` : Latest recognized sign
- `POST /clear_prediction` : Reset current prediction
- `POST /upload` : Upload video for prediction
- `POST /text_to_sign` : Text to sign lookup
- `GET /available_signs` : List indexed signs
- `POST /speak` : Text-to-speech output

## Troubleshooting

- Camera not opening:
  - Close apps already using webcam (Zoom/Teams/Camera app).
  - Verify webcam permission in Windows Privacy settings.

- TensorFlow installation issues:
  - Use Python 3.10 or 3.11 for best compatibility with current dependency versions.

- No predictions in live mode:
  - Ensure hand is visible and well-lit.
  - Wait until sequence buffer fills (30 frames).

- Text-to-sign not finding phrases:
  - Run [index_dataset.py](index_dataset.py) again after adding new media.

## Notes on Repository Size

Large raw datasets/videos are excluded via [.gitignore](.gitignore) to keep the repository lightweight. If someone needs full training data, share it separately (drive/cloud storage).

## License

Add your preferred license (MIT/Apache-2.0/etc.) in a `LICENSE` file.
