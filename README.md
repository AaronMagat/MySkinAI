# MySkinAI

MySkinAI is a web application for exploring skin-related information through image-based machine-learning experiments and UV index data.

## Features

- Image upload interface for experimental skin-image classification
- Skin information and risk-profile pages
- UV index lookup using OpenUV
- Flask web application with HTML, CSS and JavaScript

## Requirements

- Python 3.11
- The trained model file `model.h5` in the project root (or set `MODEL_PATH`)
- An OpenUV API key for UV index data

## Local setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:

   - `OPENUV_API_KEY`: required for UV index lookups
   - `MODEL_PATH`: optional path to the trained Keras model; defaults to `model.h5`
   - `PORT`: optional web server port; defaults to `5000`

4. Place the trained model at the configured model path.
5. Start the application:

   ```bash
   python app.py
   ```

Open `http://127.0.0.1:5000`. The `/health` endpoint reports whether the model loaded successfully.

## Deployment

Use the included `Procfile` with a Python hosting platform that supports Gunicorn. Configure `OPENUV_API_KEY` and ensure the model file is available at `MODEL_PATH`. Model weights should be stored using Git LFS or a secure model-artifact store, not committed as a regular Git blob.

## Important limitations

- Image classification is experimental and has not been established as a clinically validated diagnostic test.
- A model output is not a measure of diagnostic accuracy and must not be used to diagnose or rule out skin cancer.
- A qualified healthcare professional should assess any new, changing, bleeding, painful or otherwise concerning skin lesion.
- UV index lookup requires a valid OpenUV API key and network access.
