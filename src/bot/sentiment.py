import asyncio
import numpy as np
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import InputAudioRawFrame, Frame
from pipecat.frames.frames import VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from dataclasses import dataclass

_global_classifier = None

def preload_classifier():
    """Forces the HuggingFace pipeline to download and cache the model globally."""
    global _global_classifier
    if _global_classifier is None:
        from transformers import pipeline
        _global_classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
    return _global_classifier


@dataclass
class VoiceSentimentFrame(Frame):
    emotion: str

    def __post_init__(self):
        super().__post_init__()


class VoiceSentimentProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._audio_buffer = bytearray()
        self._is_speaking = False
        self._classifier = None
        self._current_sample_rate = 16000

    def _get_classifier(self):
        return preload_classifier()

    async def process_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._is_speaking = True
            self._audio_buffer.clear()
            await self.push_frame(frame, direction)

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._is_speaking = False
            if len(self._audio_buffer) > 0:
                audio_data = bytes(self._audio_buffer)
                asyncio.create_task(self._analyze_sentiment(audio_data, direction))
            await self.push_frame(frame, direction)

        elif isinstance(frame, InputAudioRawFrame):
            if self._is_speaking:
                self._audio_buffer.extend(frame.audio)
                self._current_sample_rate = frame.sample_rate
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)

    async def _analyze_sentiment(self, audio_bytes: bytes, direction: FrameDirection):
        try:
            import torch
            import torchaudio.functional as F

            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_np)

            if self._current_sample_rate != 16000:
                audio_tensor = F.resample(audio_tensor, orig_freq=self._current_sample_rate, new_freq=16000)

            audio_np_16k = audio_tensor.numpy()

            classifier = await asyncio.to_thread(self._get_classifier)
            result = await asyncio.to_thread(classifier, audio_np_16k)

            if result and len(result) > 0:
                emotion = result[0]["label"]
                await self.push_frame(VoiceSentimentFrame(emotion=emotion), direction)
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")