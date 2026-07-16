import asyncio
import numpy as np
import concurrent.futures
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import InputAudioRawFrame, Frame
from pipecat.frames.frames import VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from dataclasses import dataclass

_global_classifier = None

def _run_inference_process(audio_bytes: bytes, current_sample_rate: int):
    """Runs entirely inside a separate child process to completely bypass the GIL."""
    global _global_classifier
    if _global_classifier is None:
        from transformers import pipeline
        _global_classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
    
    import torch
    import torchaudio.functional as F

    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    audio_tensor = torch.from_numpy(audio_np)

    if current_sample_rate != 16000:
        audio_tensor = F.resample(audio_tensor, orig_freq=current_sample_rate, new_freq=16000)

    audio_np_16k = audio_tensor.numpy()
    
    result = _global_classifier(audio_np_16k)
    return result

# Single background process to prevent memory bloat, completely isolating CPU compute
_process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)

def preload_classifier():
    """Warms up the HuggingFace model inside the child process pool."""
    print("Warming up sentiment model in background process...")
    dummy_audio = bytes(32000) # 1 second of silence
    future = _process_pool.submit(_run_inference_process, dummy_audio, 16000)
    future.result() # Wait for it to finish loading and processing
    print("Sentiment model warmed up successfully.")

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
        self._current_sample_rate = 16000

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
            loop = asyncio.get_running_loop()
            
            # Offload heavy CPU inference to the isolated process pool
            result = await loop.run_in_executor(
                _process_pool, 
                _run_inference_process, 
                audio_bytes, 
                self._current_sample_rate
            )

            if result and len(result) > 0:
                emotion = result[0]["label"]
                await self.push_frame(VoiceSentimentFrame(emotion=emotion), direction)
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")