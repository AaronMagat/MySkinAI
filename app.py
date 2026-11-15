"""MySkinAI Flask application.

Image classification is experimental and must not be used as a medical diagnosis.
"""
from __future__ import annotations

import logging
import os
import tempfile
import hashlib
import threading
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

try:
    import numpy as np
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image as keras_image
except Exception as exc:  # TensorFlow may be unavailable in lightweight environments.
    load_model = None
    keras_image = None
    TENSORFLOW_IMPORT_ERROR = str(exc)
else:
    TENSORFLOW_IMPORT_ERROR = None

try:
    import geocoder
except Exception:
    geocoder = None

try:
    from geopy.geocoders import Nominatim
except Exception:
    Nominatim = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "model.h5"))
MODEL_URL = os.environ.get("MODEL_URL", "https://media.githubusercontent.com/media/Gcan3/PythonSkin/main/model.h5")
MODEL_SHA256 = "1a6482330b80cedbcf1fca7dee5e381bf45d138bb01417cadab52dd6b90e8c4a"
MODEL_SIZE = 463270040
model_lock = threading.Lock()
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("myskinai")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["TEMPLATES_AUTO_RELOAD"] = os.environ.get("FLASK_ENV") == "development"

# Model weights are downloaded and verified on the first analysis request.
model = None
MODEL_LOAD_ERROR = None


def ensure_model_loaded():
    global model, MODEL_LOAD_ERROR
    if model is not None:
        return model
    if load_model is None:
        MODEL_LOAD_ERROR = TENSORFLOW_IMPORT_ERROR or "TensorFlow is not installed"
        return None
    with model_lock:
        if model is not None:
            return model
        temp_name = None
        try:
            if not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size != MODEL_SIZE:
                MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                downloaded = 0
                with requests.get(MODEL_URL, stream=True, timeout=(20, 180)) as response:
                    response.raise_for_status()
                    with tempfile.NamedTemporaryFile(prefix="myskinai_model_", suffix=".h5", dir=str(MODEL_PATH.parent), delete=False) as temp:
                        temp_name = temp.name
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                downloaded += len(chunk)
                                if downloaded > MODEL_SIZE:
                                    raise ValueError("Downloaded model is larger than expected.")
                                digest.update(chunk)
                                temp.write(chunk)
                if downloaded != MODEL_SIZE or digest.hexdigest() != MODEL_SHA256:
                    raise ValueError("Downloaded model failed size or SHA-256 verification.")
                os.replace(temp_name, MODEL_PATH)
                temp_name = None
            model = load_model(str(MODEL_PATH))
            MODEL_LOAD_ERROR = None
            logger.info("Verified and loaded MySkinAI model")
            return model
        except Exception as exc:
            MODEL_LOAD_ERROR = str(exc)
            logger.exception("Could not download or load AI model")
            return None
        finally:
            if temp_name and os.path.exists(temp_name):
                os.remove(temp_name)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(file_path: str, loaded_model):
    if keras_image is None or loaded_model is None:
        raise RuntimeError("The image analysis model is not configured.")
    img = keras_image.load_img(file_path, target_size=(224, 224), color_mode="rgb")
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0).astype("float32") / 255.0
    prediction = np.asarray(loaded_model.predict(img_array, verbose=0))
    if prediction.size != 1:
        raise ValueError("The configured model must return one binary classification score.")
    score = float(prediction.reshape(-1)[0])
    if not np.isfinite(score) or score < 0 or score > 1:
        raise ValueError("The model returned an invalid probability score.")
    return ("Malignant" if score >= 0.5 else "Benign"), score


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_ready": model is not None,
        "model_download_configured": bool(MODEL_URL),
        "model_error": MODEL_LOAD_ERROR,
        "message": "Model is not configured. Image analysis is unavailable." if model is None else None,
    })


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/analyze-skin")
def analyze_skin():
    return render_template("analyze-skin.html", model_ready=model is not None)


@app.get("/risk-profile")
def risk_profile():
    return render_template("risk-profile.html")


@app.get("/skin-type")
def skin_type():
    return render_template("skin-type.html")


@app.post("/upload")
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "Choose an image to upload."}), 400

    uploaded = request.files["file"]
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "Choose an image to upload."}), 400
    if not allowed_file(uploaded.filename):
        return jsonify({"error": "Only PNG, JPG and JPEG images are supported."}), 400
    active_model = ensure_model_loaded()
    if active_model is None:
        return jsonify({
            "error": "Image analysis is temporarily unavailable. The AI model could not be downloaded or loaded.",
            "model_ready": False,
        }), 503

    suffix = Path(secure_filename(uploaded.filename)).suffix.lower()
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="myskinai_", suffix=suffix, delete=False) as temp:
            temp_path = temp.name
            uploaded.save(temp)
        if os.path.getsize(temp_path) == 0:
            return jsonify({"error": "The uploaded image is empty."}), 400
        result, score = predict_image(temp_path, active_model)
        return jsonify({
            "result": result,
            "scaled_value": score,
            "notice": "Experimental result only. This is not a medical diagnosis.",
        })
    except (OSError, ValueError) as exc:
        logger.info("Image upload rejected: %s", exc)
        return jsonify({"error": "The image could not be processed. Try another PNG or JPEG image."}), 400
    except Exception:
        logger.exception("Image analysis failed")
        return jsonify({"error": "Image analysis failed. Please try again later."}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning("Could not remove temporary upload %s", temp_path)


def get_user_location():
    if geocoder is None:
        return None
    try:
        result = geocoder.ip("me")
        if result and result.latlng:
            return float(result.latlng[0]), float(result.latlng[1])
    except Exception:
        logger.exception("IP-based location lookup failed")
    return None


def get_place_from_geopy(lat: float, lon: float):
    if Nominatim is None:
        return None
    try:
        locator = Nominatim(user_agent="MySkinAI/1.0 (UV index lookup)")
        place = locator.reverse((lat, lon), language="en", timeout=8)
        if not place:
            return None
        address = place.raw.get("address", {})
        city = address.get("city") or address.get("town") or address.get("village") or address.get("state")
        country = address.get("country")
        return ", ".join(part for part in (city, country) if part) or None
    except Exception:
        logger.exception("Reverse geocoding failed")
        return None


def get_uv_index(lat: float, lon: float):
    """Retrieve current and daily maximum UV index from Open-Meteo (no API key required)."""
    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "uv_index",
                "daily": "uv_index_max",
                "timezone": "auto",
                "forecast_days": 1,
            },
            timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
        current = (payload.get("current") or {}).get("uv_index")
        daily = payload.get("daily") or {}
        maxima = daily.get("uv_index_max") or []
        if current is None or not maxima or maxima[0] is None:
            return {"error": "UV data is not available for this location right now."}
        return {
            "current_uv_index": round(float(current), 1),
            "uv_max": round(float(maxima[0]), 1),
            "location": None,
        }
    except requests.RequestException:
        logger.exception("Open-Meteo UV request failed")
        return {"error": "Could not retrieve UV index data right now. Please try again later."}
    except (TypeError, ValueError, KeyError):
        logger.exception("Open-Meteo returned invalid UV data")
        return {"error": "The UV service returned invalid data."}


@app.get("/uv-index")
def uv_index():
    lat_value = request.args.get("lat")
    lon_value = request.args.get("lon")
    if lat_value is not None and lon_value is not None:
        try:
            lat, lon = float(lat_value), float(lon_value)
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError
        except ValueError:
            return render_template("uv-index.html", uv_index_data=None,
                                   location="Unknown location",
                                   error="Please provide valid latitude and longitude."), 400
    else:
        coords = get_user_location()
        if not coords:
            return render_template("uv-index.html", uv_index_data=None,
                                   location="Location unavailable",
                                   error="Allow location access or provide coordinates to view the UV index.")
        lat, lon = coords

    location = request.args.get("location") or get_place_from_geopy(lat, lon) or "Your location"
    data = get_uv_index(lat, lon)
    if "error" in data:
        return render_template("uv-index.html", uv_index_data=None, location=location,
                               error=data["error"]), 503
    return render_template("uv-index.html", uv_index_data=data,
                           location=data.get("location") or location, error=None)


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Image is too large. Maximum upload size is 10 MB."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")),
            debug=os.environ.get("FLASK_DEBUG", "0") == "1")
