# ISL Sign Language Web App

ISL Sign Language Web App is a Flask-based utility that helps bridge communication gaps by translating hand gestures into text and speech, and converting input text into available ISL visual resources (images/videos).

It combines real-time webcam inference, uploaded video inference, and text-to-sign retrieval in one web interface.

## Features

- Real-time sign recognition from webcam stream
- Sequence-based gesture classification using TensorFlow LSTM
- Text-to-Speech output for recognized signs
- Video upload and offline sign prediction
- Text-to-sign search across indexed ISL media assets
- Sign availability API for quick integrations and UI auto-complete
- Confidence thresholding and buffered prediction logic for stability

## Architecture

Single-service Python web architecture:

| Component | Tech Stack | Purpose |
|---|---|---|
| Web API + UI | Flask + Jinja2 | Serves pages and JSON endpoints |
| CV Pipeline | OpenCV + MediaPipe | Captures frames and extracts hand landmarks |
| ML Inference | TensorFlow/Keras + joblib | Loads LSTM model and label encoder for prediction |
| Dataset Indexing | Python scripts | Builds image/video lookup index for text-to-sign |
| Data Persistence | JSON + file storage | Sign index and uploaded video handling |

## Prerequisites

Install these before running:

- Python 3.10 or 3.11 (recommended for TensorFlow 2.15 compatibility)
- pip (bundled with Python)
- Webcam (for live recognition)

Verify installation:

```powershell
python --version
pip --version
```

## Quick Start

### 1. Clone the Repository

```powershell
git clone <your-repo-url>
cd sign-language-webapp
```

### 2. Create and Activate Virtual Environment

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyttsx3
```

### 4. Run the Web App

```powershell
py app.py
```

Open in browser:

- http://127.0.0.1:5000
- http://localhost:5000

## Dataset (External Download)

The full ISL corpus is intentionally not committed to this repository to keep clone/push size manageable.

Use the original dataset source link:

- Dataset URL: `https://data.mendeley.com/datasets/kcmpdxky7p/1`
- Dataset size: approximately 8 GB

Expected local folder after download and extraction:

```text
sign-language-webapp/
|-- ISL_CSLRT_Corpus/
|   |-- Frames_Word_Level/
|   |-- Videos_Sentence_Level/
|   |-- corpus_csv_files/
|   |-- ISL_CSLRT.txt
```

Important:

- Keep the folder name exactly as `ISL_CSLRT_Corpus`.
- Place it at the project root (same level as `app.py`).
- After adding dataset files, run `py index_dataset.py` to extract/copy only required sign assets into `static/sign_images/` and `static/sign_videos/`, then rebuild `sign_index.json`.

If you only want inference with pre-trained files, you can run the app without downloading the full raw dataset.

## Model and Dataset Workflow

Use these scripts when you want to prepare data or retrain:

Recommended order for a fresh setup with new raw data:

1. Download and extract dataset into `ISL_CSLRT_Corpus/`
2. Run `py index_dataset.py`
3. (Optional) Run `py collect_data.py` for custom gestures
4. (Optional) Run `py train_model.py` to retrain
5. Run `py app.py`

### Collect Training Data

```powershell
py collect_data.py
```

What it does:
- Captures 30-frame gesture sequences
- Extracts normalized two-hand landmarks (126 features/frame)
- Stores flattened sequences in `sign_sequence_data.csv`
- Optionally applies augmentation

### Train Model

```powershell
py train_model.py
```

Outputs:
- `sign_lstm_model.keras`
- `best_model_checkpoint.keras`
- `label_encoder.joblib`
- `training_history.png`

### Build Text-to-Sign Index

```powershell
py index_dataset.py
```

Outputs:
- `sign_index.json`
- Media copies into `static/sign_images/` and `static/sign_videos/`

### Optional Standalone Real-Time Predictor

```powershell
py real_time_predict.py
```

## Runtime Configuration

Important defaults in `app.py`:

- Model file: `sign_lstm_model.keras`
- Encoder file: `label_encoder.joblib`
- Index file: `sign_index.json`
- Upload folder: `uploads/`
- Frames per sample: 30
- Features per frame: 126
- Confidence threshold: 0.75
- Server host/port: `0.0.0.0:5000`

## API Endpoints

All routes are served from the same Flask app.

### Core

- `GET /` : Main web UI
- `GET /health` : Status of model, camera, and indexed signs

### Live Recognition

- `POST /start_camera` : Initializes webcam stream
- `POST /stop_camera` : Stops webcam stream
- `GET /video_feed` : MJPEG stream for live frames
- `GET /current_prediction` : Latest stable prediction object
- `POST /clear_prediction` : Resets prediction state

### Speech

- `POST /speak` : Speaks supplied text using local TTS engine

Payload:

```json
{
  "text": "hello"
}
```

### Upload-Based Prediction

- `POST /upload` : Uploads a video and returns translated signs

Form-data key:
- `video`

### Text-to-Sign

- `POST /text_to_sign` : Maps text words/phrases to indexed sign media
- `GET /available_signs` : Returns all indexed sign keys

`/text_to_sign` payload example:

```json
{
  "text": "how are you"
}
```

## Project Structure

```text
sign-language-webapp/
|-- app.py
|-- collect_data.py
|-- train_model.py
|-- real_time_predict.py
|-- index_dataset.py
|-- requirements.txt
|-- sign_lstm_model.keras
|-- label_encoder.joblib
|-- sign_index.json
|-- templates/
|   |-- index.html
|-- static/
|   |-- sign_images/
|   |-- sign_videos/
|-- uploads/
|-- ISL_CSLRT_Corpus/
```

## Troubleshooting

### Backend/App Startup

- `ModuleNotFoundError`:
  - Activate virtual environment and reinstall dependencies.

- `pyttsx3` import error:
  - Install it manually with `pip install pyttsx3`.

- TensorFlow install issues on newest Python:
  - Use Python 3.10/3.11 virtual environment.

### Camera and Live Predictions

- Camera does not open:
  - Close other apps using webcam (Zoom/Teams/Camera app).
  - Check Windows privacy camera permissions.

- Predictions remain "Detecting...":
  - Keep hand visible for enough frames.
  - Improve lighting and reduce background clutter.

- Wrong/unstable predictions:
  - Retrain with more samples per class.
  - Ensure gesture consistency during collection.

### Text-to-Sign

- Missing signs in output:
  - Confirm dataset is present under `ISL_CSLRT_Corpus/` with expected subfolders.
  - Run `py index_dataset.py` after adding new data.
  - Verify files exist under `static/sign_images/` or `static/sign_videos/`.

## Security and Deployment Notes

- Current setup is intended for local/dev usage.
- TTS executes locally on the host machine.
- Uploaded media is stored in local `uploads/`.

For production, consider:

- Running behind HTTPS reverse proxy
- Restricting upload size and file types
- Adding authentication/authorization
- Storing uploads and models in managed storage

## Repository Size Guidance

Large raw datasets and videos are excluded by `.gitignore` to keep cloning/pushing manageable. Share full corpora separately (cloud storage, release artifacts, or dataset mirror).

## Documentation Maintenance Checklist

When changing behavior, update docs in the same PR:

- If API routes change, update endpoint sections in this README.
- If model files or training flow changes, update workflow and output filenames.
- If startup commands change, update Quick Start.
- If dependencies change, update prerequisites and install commands.
- If folder structure changes, update the project tree.
- Update this README date stamp after docs sync.

## Documentation Status

Last updated: 2026-04-19

## Dataset Citation

If your dataset source has a required citation/license statement, add it here before public release.


