import numpy as np
import librosa
from pydub import AudioSegment
from io import BytesIO
import soundfile as sf
import webrtcvad
import io

def decode_audio_to_16k_mono(file_bytes: io.BytesIO, filename: str | None = None):
    """
    優先用 soundfile 一次讀 → 轉單聲道 → 16k。
    若不支援（常見 webm），用 pydub+ffmpeg 轉 WAV 再讀
    回傳: (y: float32, sr=16000)
    """
    file_bytes.seek(0)
    try:
        # 首選：soundfile 直接讀成 float32
        y, sr = sf.read(file_bytes, dtype='float32', always_2d=False)
    except Exception:
        # soundfile 讀失敗（可能 webm），用 pydub 轉
        file_bytes.seek(0)
        fmt = "webm" if (filename and filename.lower().endswith(".webm")) else None
        audio = AudioSegment.from_file(file_bytes, format=fmt) if fmt else AudioSegment.from_file(file_bytes)
        wav_io = BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        y, sr = sf.read(wav_io, dtype='float32', always_2d=False)

    # 轉單聲道
    if y.ndim > 1:
        y = np.mean(y, axis=1)
        
    # 重取樣到 16k
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000, res_type="kaiser_fast")
        sr = 16000
    return y.astype(np.float32), sr

class WebRtcVad:
    """
    以 WebRTC VAD 取代 Silero（避免 Windows 上 torch/DLL 問題）
    先偵測語音的時間區段（timestamps），只對語音區段切窗與特徵
    介面維持 speech_timestamps(y) -> [(start, end)]（樣本索引）
    """
    def __init__(self, sr=16000, aggressiveness=2, frame_ms=20):
        self.sr = sr
        self.vad = webrtcvad.Vad(aggressiveness)  # aggressiveness: 0~3，越大越嚴格
        self.frame_ms = frame_ms                   # WebRTC VAD 只能處理 10/20/30ms 的短幀（16-bit PCM）

    def _float_to_int16(self, y: np.ndarray) -> bytes:
        y_clip = np.clip(y, -1.0, 1.0)
        pcm16 = (y_clip * 32767.0).astype(np.int16)
        return pcm16.tobytes()

    def speech_timestamps(self, y: np.ndarray, min_speech_s=0.3, max_silence_s=0.5):
        """
        先把 float32 轉成 int16 PCM bytes，依序切成固定長度的 frame。
        用 VAD 判斷每個 frame 是否為語音，將連續語音 frame 併成一段。
        容許短暫靜音（max_silence_s）仍屬同一段，避免說話中間停頓被切斷。
        若抓不到語音，最後退回 energy-based（librosa.effects.split），避免空結果。
        """
        pcm_bytes = self._float_to_int16(y)

        bytes_per_sample = 2
        samples_per_frame = int(self.sr * (self.frame_ms / 1000.0))
        frame_bytes = samples_per_frame * bytes_per_sample

        frames = []
        for i in range(0, len(pcm_bytes) - frame_bytes + 1, frame_bytes):
            frame = pcm_bytes[i:i + frame_bytes]
            is_speech = self.vad.is_speech(frame, self.sr)
            frames.append(is_speech)

        ts_list = []
        in_speech = False
        start_idx = 0
        silence_run = 0
        max_silence_frames = int(max_silence_s / (self.frame_ms / 1000.0))
        min_speech_frames = int(min_speech_s / (self.frame_ms / 1000.0))

        for idx, flag in enumerate(frames):
            # 目前是語音
            if flag: 
                if not in_speech:
                    in_speech = True
                    start_idx = idx
                silence_run = 0
            # 目前是靜音
            else:    
                if in_speech:
                    silence_run += 1
                    # 靜音累積超過上限 → 前一段語音結束
                    if silence_run > max_silence_frames:
                        end_idx = idx - silence_run
                        if end_idx - start_idx >= min_speech_frames:
                            s = start_idx * samples_per_frame
                            e = end_idx * samples_per_frame
                            ts_list.append((s, e))
                        in_speech = False
                        silence_run = 0
                        
        # 收尾：若最後仍在語音狀態就補上一段
        if in_speech:
            end_idx = len(frames)
            if end_idx - start_idx >= min_speech_frames:
                s = start_idx * samples_per_frame
                e = end_idx * samples_per_frame
                ts_list.append((s, e))

        # 若抓不到語音，退回 用音量大小切，避免空結果
        if not ts_list:
            intervals = librosa.effects.split(y, top_db=30)
            ts_list = [(int(s), int(e)) for s, e in intervals]

        return ts_list

def window_slices_from_ts(y, sr, ts_list, win_s=3.0, hop_s=1.5, max_windows_per_ts=999):
    """
    片段長度固定、重疊 50%（3s / 1.5s）   
    """
    win = int(win_s * sr)
    hop = int(hop_s * sr)
    slices = []
    for (s, e) in ts_list:
        start = s
        count = 0
        # 覆疊切窗
        while start + win <= e and count < max_windows_per_ts:
            slices.append((start, start + win))
            start += hop
            count += 1
        # 視情況補尾窗：避免在語音尾端剛好少一點點而沒有被取到
        if e - win > s and (e - win) - (start - hop) > (0.3 * win):
            slices.append((e - win, e))
    return slices

def extract_features_batch(y_list, sr=16000, max_len=500):
    """把多段 y 做成一個 batch 特徵：(B, 500, feat_dim)"""
    feats = []
    n_fft = 512
    hop_length = 128
    for y in y_list:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=15, n_fft=n_fft, hop_length=hop_length)
        delta_mfcc = librosa.feature.delta(mfcc)
        log_mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=64))
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)

        min_frames = min(mfcc.shape[1], delta_mfcc.shape[1], log_mel.shape[1], rms.shape[1], zcr.shape[1])
        f = np.vstack([mfcc[:, :min_frames], delta_mfcc[:, :min_frames], log_mel[:, :min_frames], rms[:, :min_frames], zcr[:, :min_frames]])
        if f.shape[1] < max_len:
            f = np.pad(f, ((0, 0), (0, max_len - f.shape[1])), mode='constant')
        else:
            f = f[:, :max_len]
        feats.append(f.T)  # (500, feat_dim)
    X = np.stack(feats, axis=0)
    return X