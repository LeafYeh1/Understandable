import os
import hashlib
import pathlib
import requests
import numpy as np
from tensorflow.keras.models import load_model
import whisper
from .audio_processor import extract_features_batch, window_slices_from_ts, WebRtcVad
from ..models import class_names 

# 全域變數
_emotion_model = None
_whisper_model = None

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest().lower()

# 模型下載 / 檢查
def ensure_model(logger, model_path, model_url, model_sha):
    if os.path.exists(model_path):
        if model_sha:
            try:
                if _sha256(model_path) == model_sha:
                    logger.info("[model] cached ok: %s", model_path)
                    return
                else:
                    logger.warning("[model] sha mismatch, re-downloading")
            except Exception as e:
                logger.warning("[model] checksum error: %r, re-downloading", e)
        else:
            logger.info("[model] cached (no sha): %s", model_path)
            return

    if not model_url:
        raise RuntimeError("EMOTION_MODEL_URL not set and model file missing")

    logger.info("[model] downloading: %s", model_url)
    r = requests.get(model_url, timeout=120)  
    r.raise_for_status()
    pathlib.Path(model_path).write_bytes(r.content)
    logger.info("[model] saved: %s (%d bytes)", model_path, len(r.content))

    if model_sha and _sha256(model_path) != model_sha:
        raise RuntimeError("Model SHA256 mismatch after download")
    logger.info("[model] ready.")

# 檢查全域變數，下載並載入模型
def get_emotion_model(logger, config):
    global _emotion_model 
    if _emotion_model is None:
        ensure_model(logger, config['MODEL_PATH'], config['MODEL_URL'], config['MODEL_SHA256'])
        logger.info("[model] loading %s ...", config['MODEL_PATH'])
        _emotion_model = load_model(config['MODEL_PATH'], compile=False)
        logger.info("[model] loaded.")
    return _emotion_model

def get_whisper(logger, config):
    global _whisper_model
    if _whisper_model is None:
        size = config.get('WHISPER_MODEL', 'tiny')
        _whisper_model = whisper.load_model(size)
    return _whisper_model

class EmotionAnalyzer:
    def __init__(self, logger, config, sr=16000, win_s=3.0, hop_s=1.5):
        self.logger = logger
        self.config = config
        self.sr = sr
        self.win_s = win_s
        self.hop_s = hop_s
        self.vad = WebRtcVad(sr=sr, aggressiveness=2, frame_ms=20)

    def analyze(self, y_all: np.ndarray):
        """
        1) WebRTC VAD 找語音區段
        2) 各區段滑動窗（預設 win=3s, hop=1.5s）
        3) 批次特徵 → 一次 model.predict
        4) 回傳：({emotion: count}, [label序列])
        """
        model = get_emotion_model(self.logger, self.config)
        ts_list = self.vad.speech_timestamps(y_all)
        if not ts_list:
            return {c: 0 for c in class_names}, []

        slices = window_slices_from_ts(y_all, self.sr, ts_list, self.win_s, self.hop_s)
        win_samps = int(self.win_s * self.sr)
        chunks = [y_all[s:e] for (s, e) in slices if e - s == win_samps]
        if not chunks:
            return {c: 0 for c in class_names}, []

        X = extract_features_batch(chunks, sr=self.sr, max_len=500)
        preds = model.predict(X, verbose=0)  
        labels_idx = np.argmax(preds, axis=1)
        labels = [class_names[i] for i in labels_idx]

        counts = {c: 0 for c in class_names}
        for lb in labels:
            counts[lb] += 1
        return counts, labels

def preload_models(logger, app_config):
    """
    在應用程式啟動時，預先載入所有耗時的模型。
    """
    logger.info("[Preload] 正在預先載入模型...")
    try:
        get_emotion_model(logger, app_config)
        get_whisper(logger, app_config)
        logger.info("[Preload] 所有模型載入完畢！")
    except Exception as e:
        logger.error("[Preload] 模型預載失敗: %s", e)