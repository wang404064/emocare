"""
AudioEmotionRecognizer — 语音情绪识别 (ASR + SER)
基于阿里 SenseVoiceSmall，单模型同时输出文本转录 + 情绪标签 + 音频事件。

流程:
  音频 (WAV/bytes) → torchaudio 预处理 (16kHz mono) → VAD 静音裁剪
  → SenseVoice 推理 → 解析情感标签 + 音频事件 + 纯文本
  → 映射到 EmoCare 7 维情绪向量

依赖: funasr, torchaudio, soundfile
"""
import re
import io
import numpy as np
from typing import Dict, Tuple, Optional
from loguru import logger

from ..core.config import settings

# SenseVoice 情感标签 → EmoCare 7 维情绪
EMOTION_TAG_MAP = {
    "HAPPY":     "joy",
    "SAD":       "sadness",
    "ANGRY":     "anger",
    "NEUTRAL":   "calm",
    "FEARFUL":   "anxiety",
    "DISGUSTED": "anger",
    "SURPRISED": "anxiety",
}

# 音频事件 → 情绪增量
AUDIO_EVENT_BOOST = {
    "CRY":      {"sadness": 0.3, "hopelessness": 0.2},
    "LAUGHTER": {"joy": 0.3},
    "LAUGH":    {"joy": 0.25},
    "SOB":      {"sadness": 0.25},
    "SIGH":     {"sadness": 0.15},
    "SCREAM":   {"anxiety": 0.3, "anger": 0.2},
    "APPLAUSE": {},
    "BGM":      {},
    "SILENCE":  {},
}

# 情感标签正则
_EMOTION_TAG_RE = re.compile(
    r"<\|(HAPPY|SAD|ANGRY|NEUTRAL|FEARFUL|DISGUSTED|SURPRISED)\|>"
)
_AUDIO_EVENT_RE = re.compile(
    r"<\|(CRY|LAUGHTER|LAUGH|SOB|SIGH|SCREAM|APPLAUSE|BGM|SILENCE)\|?>"
)
_LANG_TAG_RE = re.compile(r"<\|(zh|en|yue|ja|ko|nospeech)\|>")

# EmoCare 7 维顺序
EMOTION_ORDER = [
    "sadness", "anxiety", "anger", "loneliness",
    "hopelessness", "calm", "joy"
]


class AudioEmotionRecognizer:
    """语音情绪识别器 — SenseVoiceSmall 封装"""

    def __init__(self, device: Optional[str] = None):
        self._model = None
        self._loaded = False
        self._device = device or settings.EMOTION_MODEL_DEVICE or (
            "cuda" if _cuda_available() else "cpu"
        )

    def _load_model(self):
        if self._loaded:
            return
        try:
            from funasr import AutoModel

            logger.info(
                f"加载 SenseVoiceSmall 模型 (device={self._device})..."
            )
            self._model = AutoModel(
                model=settings.AUDIO_MODEL_NAME,
                vad_model=getattr(settings, "AUDIO_VAD_MODEL", "fsmn-vad"),
                vad_kwargs={"max_single_segment_time": settings.AUDIO_MAX_DURATION_SEC * 1000},
                device=self._device,
                disable_update=True,
            )
            self._loaded = True
            logger.info("SenseVoiceSmall 加载成功")
        except ImportError:
            logger.error("未安装 funasr，请执行: pip install funasr")
            raise
        except Exception as e:
            logger.error(f"SenseVoiceSmall 加载失败: {e}")
            raise

    def preprocess_audio(
        self, audio_data: bytes, target_sr: int = 16000
    ) -> Tuple[np.ndarray, int]:
        """
        音频预处理: 解码 → 重采样 → 单声道 → VAD 裁剪。

        Returns:
            (audio_array, sample_rate)
        """
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError("未安装 soundfile，请执行: pip install soundfile")

        # 从 bytes 解码
        audio_np, orig_sr = sf.read(io.BytesIO(audio_data), dtype="float32")

        # 多声道 → 单声道
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)

        # 重采样到 16kHz
        if orig_sr != target_sr:
            audio_np = self._resample(audio_np, orig_sr, target_sr)

        # 归一化
        peak = np.abs(audio_np).max()
        if peak > 0:
            audio_np = audio_np / peak * 0.95

        # 简易 VAD：裁剪首尾静音 (能量阈值 -40dB)
        audio_np = self._trim_silence(audio_np, threshold_db=-40)

        return audio_np.astype(np.float32), target_sr

    def _resample(
        self, audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """简易线性重采样（避免依赖 torchaudio 的复杂 API）"""
        if orig_sr == target_sr:
            return audio
        try:
            import torchaudio.functional as F
            import torch
            tensor = torch.from_numpy(audio).float().unsqueeze(0)
            resampled = F.resample(tensor, orig_freq=orig_sr, new_freq=target_sr)
            return resampled.squeeze(0).numpy()
        except Exception:
            # fallback: scipy
            from scipy.signal import resample_poly
            import fractions
            ratio = fractions.Fraction(target_sr, orig_sr)
            return resample_poly(audio, ratio.numerator, ratio.denominator)

    def _trim_silence(
        self, audio: np.ndarray, threshold_db: float = -40
    ) -> np.ndarray:
        """裁剪首尾静音段"""
        if len(audio) < 160:  # <10ms 不处理
            return audio

        # 分帧算能量
        frame_len = 320  # 20ms @16kHz
        num_frames = len(audio) // frame_len
        if num_frames < 2:
            return audio

        energy = np.array([
            np.sum(audio[i * frame_len:(i + 1) * frame_len] ** 2)
            for i in range(num_frames)
        ])
        energy_db = 10 * np.log10(energy + 1e-10)
        threshold_linear = 10 ** (threshold_db / 10)

        # 找第一个和最后一个有效帧
        active = energy > threshold_linear
        if not active.any():
            return audio  # 全是静音，保留

        first = active.argmax()
        last = len(active) - active[::-1].argmax() - 1

        # 保留前后各 100ms padding
        pad = 5  # 5帧 = 100ms
        start = max(0, first - pad) * frame_len
        end = min(len(audio), (last + pad + 1) * frame_len)
        return audio[start:end]

    def recognize(self, audio_data: bytes) -> Dict:
        """
        语音情绪识别主入口。

        Args:
            audio_data: 原始音频 bytes (WAV/FLAC/OGG 等)

        Returns:
            {
                "text": "转录文本（纯净，无标签）",
                "audio_emotion": "主要音频情绪",
                "audio_emotion_intensity": 0.82,
                "audio_emotion_vector": [0.0, 0.1, ...],  # 7 维
                "audio_events": ["CRY", "SIGH"],
                "raw_transcript": "原始带标签文本"
            }
        """
        if not self._loaded:
            self._load_model()

        if not audio_data or len(audio_data) < 800:  # <50ms @16kHz
            return self._silence_result()

        try:
            # 预处理
            audio_np, sr = self.preprocess_audio(audio_data)

            if len(audio_np) < 160:  # <10ms
                return self._silence_result()

            # SenseVoice 推理
            result = self._model.generate(
                input=audio_np,
                cache={},
                language="zh",
                use_itn=True,
                batch_size_s=30,
            )

            if not result or len(result) == 0:
                return self._silence_result()

            raw_text = result[0].get("text", "")
            logger.debug(f"SenseVoice 原始输出: {raw_text[:200]}")

            # 解析
            return self._parse_output(raw_text)

        except Exception as e:
            logger.error(f"语音情绪识别失败: {e}")
            return self._silence_result()

    def _parse_output(self, raw_text: str) -> Dict:
        """解析 SenseVoice 输出：提取情绪标签、音频事件、纯文本"""
        # 提取情绪标签
        emotion_tags = _EMOTION_TAG_RE.findall(raw_text)
        audio_events = _AUDIO_EVENT_RE.findall(raw_text)

        # 提取纯文本（去掉所有标签）
        clean_text = raw_text
        clean_text = _LANG_TAG_RE.sub("", clean_text)
        clean_text = _EMOTION_TAG_RE.sub("", clean_text)
        clean_text = _AUDIO_EVENT_RE.sub("", clean_text)
        clean_text = " ".join(clean_text.split())

        # 构建 7 维情绪向量
        vector = self._build_emotion_vector(emotion_tags, audio_events)

        # 主情绪
        primary_idx = int(np.argmax(vector))
        primary_emotion = EMOTION_ORDER[primary_idx]
        primary_intensity = float(vector[primary_idx])

        return {
            "text": clean_text,
            "audio_emotion": primary_emotion,
            "audio_emotion_intensity": min(primary_intensity * 1.2, 1.0),
            "audio_emotion_vector": [round(float(v), 4) for v in vector],
            "audio_events": audio_events,
            "emotion_tags": emotion_tags,
            "raw_transcript": raw_text,
        }

    def _build_emotion_vector(
        self, emotion_tags: list, audio_events: list
    ) -> np.ndarray:
        """将 SenseVoice 情感标签 + 音频事件映射为 7 维情绪向量"""
        vector = np.zeros(7, dtype=np.float32)

        if not emotion_tags:
            vector[EMOTION_ORDER.index("calm")] = 0.5
            return vector

        # 1. 统计情绪标签频次，构造基础分布
        tag_counts = {}
        for tag in emotion_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        total = len(emotion_tags)
        for tag, count in tag_counts.items():
            emo_key = EMOTION_TAG_MAP.get(tag)
            if emo_key and emo_key in EMOTION_ORDER:
                idx = EMOTION_ORDER.index(emo_key)
                vector[idx] = count / total

        # 2. 音频事件加权
        for event in audio_events:
            boosts = AUDIO_EVENT_BOOST.get(event, {})
            for emo_key, boost_val in boosts.items():
                if emo_key in EMOTION_ORDER:
                    idx = EMOTION_ORDER.index(emo_key)
                    vector[idx] = min(vector[idx] + boost_val, 1.0)

        # 3. 归一化（softmax 风格，保持相对关系）
        if vector.sum() > 0:
            vector = vector / vector.sum()

        return vector

    def _silence_result(self) -> Dict:
        return {
            "text": "",
            "audio_emotion": "calm",
            "audio_emotion_intensity": 0.1,
            "audio_emotion_vector": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "audio_events": [],
            "emotion_tags": [],
            "raw_transcript": "",
        }


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# 全局单例
_audio_recognizer: Optional[AudioEmotionRecognizer] = None


def get_audio_recognizer() -> AudioEmotionRecognizer:
    global _audio_recognizer
    if _audio_recognizer is None:
        _audio_recognizer = AudioEmotionRecognizer()
    return _audio_recognizer
