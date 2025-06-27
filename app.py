from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import json
import librosa
import os

app = Flask(__name__)

# 載入模型
model = load_model("emotion_model.h5")
class_names = ['ang', 'dis', 'fear', 'happy', 'neu', 'sad']

def extract_features(file_path, max_len=500):
    y, sr = librosa.load(file_path, sr=16000)
    
    # 基本參數
    n_fft = 512
    hop_length = 128
    n_mels = 64

    # log-Mel
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec)  # (64, T)

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)  # (1, T)

    # RMS energy
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)  # (1, T)

    # 對齊時間軸（確保時間維相同）
    min_frames = min(log_mel_spec.shape[1], zcr.shape[1], rms.shape[1])
    log_mel_spec = log_mel_spec[:, :min_frames]
    zcr = zcr[:, :min_frames]
    rms = rms[:, :min_frames]

    # 合併所有特徵（垂直 stack）
    features = np.vstack([log_mel_spec, zcr, rms])  # shape: (66, T)

    # Padding or truncating to max_len
    if features.shape[1] < max_len:
        pad_width = max_len - features.shape[1]
        features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        features = features[:, :max_len]

    return features.T  # shape: (max_len, 66)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    features = extract_features(file_path)
    input_data = np.array(features).reshape(1, 500, 66)
    prediction = model.predict(input_data)
    predicted_class = class_names[np.argmax(prediction)]

    os.remove(file_path)
    print(f"預測情緒：{predicted_class}")
    print(json.dumps({"emotion": predicted_class}))
    return jsonify({"emotion": predicted_class}), 200
    


CORS(app, origins=["http://localhost:8000"])
if __name__ == "__main__":
    app.run(debug=True)

