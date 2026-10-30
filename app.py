from flask import Flask, request, jsonify, send_from_directory, session
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import io, os, sqlite3, secrets
import cv2
import base64
import csv as csv_module
import datetime
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__, static_folder=".")

# Use a fixed key via environment variable in production so sessions survive a restart.
# For local/demo use, a random key each run is fine (everyone just has to log in again
# after you restart the server).
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(16))

# ---------------- Model + species info ----------------
MODEL_PATH = "best_model.h5"
INFO_CSV = "medicinal_info.csv"
LAST_CONV_LAYER = "Conv_1"  # check model.summary() if Grad-CAM errors out; MobileNetV2 usually uses this or 'out_relu'

model = tf.keras.models.load_model(MODEL_PATH)
info_df = pd.read_csv(INFO_CSV)
CLASS_NAMES = sorted(info_df['plant_name'].tolist())  # must match train_gen.class_indices order
IMG_SIZE = (224, 224)


def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


# ---------------- User accounts (SQLite) ----------------
DB_PATH = "users.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL
    )""")
    conn.commit()
    conn.close()

init_db()


@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify({"error": "email and password required"}), 400

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name, email, generate_password_hash(password))
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "an account with that email already exists"}), 409
    conn.close()

    session["user_email"] = email
    session["user_name"] = name or email.split("@")[0]
    return jsonify({"email": email, "name": session["user_name"]})


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT name, password_hash FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()

    if not row or not check_password_hash(row[1], password):
        return jsonify({"error": "invalid email or password"}), 401

    session["user_email"] = email
    session["user_name"] = row[0] or email.split("@")[0]
    return jsonify({"email": email, "name": session["user_name"]})


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


# ---------------- Prediction ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    file = request.files["file"]
    img_array = preprocess(file.read())

    preds = model.predict(img_array)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx]) * 100
    species = CLASS_NAMES[idx]

    row = info_df[info_df["plant_name"] == species]

    def field(col):
        return row.iloc[0][col] if not row.empty and col in row.columns else ""

    return jsonify({
        "species": species,
        "confidence": round(confidence, 1),
        "use": field("use"),
        "precautions": field("precautions"),
        "preparation": field("preparation"),
        "scientific_name": field("scientific_name"),
        "hindi_name": field("hindi_name"),
        "malayalam_name": field("malayalam_name")
    })


# ---------------- User feedback / active learning ----------------
# When someone marks a prediction wrong and tells us the real species, we save
# the photo into a folder already named after the CORRECT species -- so later
# you can literally copy these folders straight into final_dataset/ and
# retrain, no relabeling needed.
FEEDBACK_DIR = "feedback_images"
FEEDBACK_LOG = "feedback_log.csv"
os.makedirs(FEEDBACK_DIR, exist_ok=True)


def log_feedback(predicted, corrected, confidence, saved_path, was_correct):
    file_exists = os.path.isfile(FEEDBACK_LOG)
    with open(FEEDBACK_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv_module.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "predicted", "corrected", "confidence", "was_correct", "image_path"])
        writer.writerow([datetime.datetime.now().isoformat(), predicted, corrected, confidence, was_correct, saved_path])


@app.route("/feedback", methods=["POST"])
def feedback():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    file = request.files["file"]
    predicted = request.form.get("predicted", "")
    corrected = request.form.get("corrected", predicted)
    confidence = request.form.get("confidence", "")
    was_correct = (corrected == predicted)

    species_folder = os.path.join(FEEDBACK_DIR, corrected.replace(" ", "_"))
    os.makedirs(species_folder, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ext = os.path.splitext(file.filename or "")[1] or ".jpg"
    saved_path = os.path.join(species_folder, f"{timestamp}_pred-{predicted.replace(' ', '_')}{ext}")
    file.save(saved_path)

    log_feedback(predicted, corrected, confidence, saved_path, was_correct)

    return jsonify({"ok": True, "was_correct": was_correct})


# ---------------- Grad-CAM ----------------
# Your model is Sequential([base_model, GlobalAveragePooling2D, Dense, Dropout, Dense]),
# so the last conv layer lives INSIDE the nested base_model, not on the outer model.
# model.get_layer(...) on the outer model can't see it directly, so we build two
# sub-models instead: one up to the last conv layer, one for everything after it.
def find_last_conv_layer(base_model):
    for layer in reversed(base_model.layers):
        if len(layer.output_shape) == 4:  # (batch, h, w, channels) -> a conv-type layer
            return layer.name
    raise ValueError("No 4D (convolutional) layer found in the base model.")


def make_gradcam_heatmap(img_array, model):
    base_model = model.layers[0]  # the nested MobileNetV2 (or similar) submodel

    try:
        last_conv_layer = base_model.get_layer(LAST_CONV_LAYER)
    except ValueError:
        auto_name = find_last_conv_layer(base_model)
        last_conv_layer = base_model.get_layer(auto_name)

    conv_model = tf.keras.Model(base_model.input, last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer in model.layers[1:]:  # everything after the base model: pooling, dense, etc.
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        conv_output = conv_model(img_array)
        tape.watch(conv_output)
        preds = classifier_model(conv_output)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


@app.route("/gradcam", methods=["POST"])
def gradcam():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    raw = request.files["file"].read()
    img_array = preprocess(raw)

    try:
        heatmap = make_gradcam_heatmap(img_array, model)
    except Exception as e:
        return jsonify({"error": f"Grad-CAM failed: {type(e).__name__}: {e}"}), 500

    orig = Image.open(io.BytesIO(raw)).convert("RGB").resize(IMG_SIZE)
    orig_np = np.array(orig)

    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (orig_np * 0.6 + heatmap_color * 0.4).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)

    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

    return jsonify({"heatmap": f"data:image/png;base64,{encoded}"})


# ---------------- Serve the UI ----------------
@app.route("/")
def serve_ui():
    return send_from_directory(".", "plant_id_ui.html")


if __name__ == "__main__":
    # use_reloader=False avoids Flask auto-restarting (and re-importing TensorFlow,
    # which takes ~20-30s) every time it notices a file change -- one clean startup instead.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True, use_reloader=False)