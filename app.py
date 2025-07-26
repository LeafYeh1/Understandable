from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import librosa
from pydub import AudioSegment
from io import BytesIO

app = Flask(__name__)
CORS(app, origins=["http://localhost:8000"])

# 定義情緒類別
class_names = ['ang', 'dis', 'fear', 'happy', 'neu', 'sad']

class EmotionAnalyzer:
    def __init__(self, model_path="emotion_model.h5"):
        self.model = load_model(model_path)
        self.sr = 16000  # 取樣率
        self.segment_duration = 2  # 每段切割長度（秒）

    def extract_features(self, y):
        """
        從音訊 array 中提取 MFCC、Delta、Chroma、Contrast 特徵並整理成固定形狀
        """
        n_fft = 512
        hop_length = 128

        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=15, n_fft=n_fft, hop_length=hop_length)
        delta_mfcc = librosa.feature.delta(mfcc, order=1)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr, n_fft=n_fft, hop_length=hop_length)
        contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr, n_fft=n_fft, hop_length=hop_length)

        min_frames = min(mfcc.shape[1], delta_mfcc.shape[1], delta2_mfcc.shape[1], chroma.shape[1], contrast.shape[1])

        # 裁切時間軸長度一致
        features = np.vstack([
            mfcc[:, :min_frames],
            delta_mfcc[:, :min_frames],
            delta2_mfcc[:, :min_frames],
            chroma[:, :min_frames],
            contrast[:, :min_frames],
        ])

        # 統一長度
        max_len = 500
        if features.shape[1] < max_len:
            pad_width = max_len - features.shape[1]
            features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
        else:
            features = features[:, :max_len]

        return features.T  # shape = (500, 64)

    def classify_segment(self, y, threshold_energy=0.01, min_duration_speech=0.5):
        """
        判斷該段音訊是靜音(0)、不確定(1)、語音(2)
        """
        if len(y) < self.sr * self.segment_duration:
            y = np.pad(y, (0, self.sr * self.segment_duration - len(y)))

        energy = np.sum(y ** 2) / len(y)
        if energy < threshold_energy:
            return 0

        voiced = sum([(e-s)/self.sr for s, e in librosa.effects.split(y, top_db=30)])
        if voiced >= min_duration_speech:
            return 2
        return 1

    def analyze(self, file_stream):
        """
        主邏輯處理流程：切割 -> 標註 -> 黏合 -> 預測 -> 統計
        """
        audio = AudioSegment.from_file(file_stream, format="wav")
        segment_ms = self.segment_duration * 1000
        segments = [audio[i:i+segment_ms] for i in range(0, len(audio), segment_ms)]

        # 對每段分類（靜音、不確定、語音）
        labels = [self.classify_segment(np.array(seg.get_array_of_samples()).astype(np.float32) / (2**15)) for seg in segments]

        # 修正不確定的段落標籤
        for i in range(len(labels)):
            if labels[i] == 1:
                if (i > 0 and labels[i-1] == 2) or (i < len(labels)-1 and labels[i+1] == 2):
                    labels[i] = 2
                elif (i == 0 or labels[i-1] == 0) and (i == len(labels)-1 or labels[i+1] == 0):
                    labels[i] = 0

        emotion_counts = {name: 0 for name in class_names}
        line_results = []

        # 找出連續語音段（4 秒）
        for i in range(len(labels) - 1):
            if labels[i] == labels[i+1] == 2:
                combined = segments[i] + segments[i+1]
                y = np.array(combined.get_array_of_samples()).astype(np.float32) / (2**15)
                features = self.extract_features(y)
                input_data = np.expand_dims(features, axis=0)
                prediction = self.model.predict(input_data)[0]
                predicted_label = class_names[int(np.argmax(prediction))]
                emotion_counts[predicted_label] += 1
                line_results.append(predicted_label)

        return emotion_counts, line_results


analyzer = EmotionAnalyzer()


@app.route("/predict", methods=["POST"])
def predict():
    """
    接收音檔並回傳情緒分析結果（餅圖 + 折線圖資料）
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    # 直接從記憶體處理檔案，不儲存到硬碟
    file_stream = BytesIO(file.read())

    pie_chart, line_chart = analyzer.analyze(file_stream)

    return jsonify({
        "pie_chart": pie_chart,
        "line_chart": line_chart
    }), 200


if __name__ == "__main__":
    app.run(debug=True)
