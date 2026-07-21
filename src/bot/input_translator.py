import os
import aiohttp
import asyncio
from loguru import logger
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import Frame, TranscriptionFrame, TTSUpdateSettingsFrame
from pipecat.services.sarvam.tts import SarvamTTSService

class InputTranslationProcessor(FrameProcessor):
    """
    Intercepts native-script TranscriptionFrames from the STT and identifies the spoken language 
    for the TTS output. It intentionally DOES NOT translate the text, passing native text 
    directly to the LLM to save latency, relying on the LLM's implicit multilingualism.
    """
    def __init__(self, customer_profile: dict):
        super().__init__()
        self.customer_profile = customer_profile
        self.api_key = os.getenv("SARVAM_API_KEY")
        
    async def process_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        await super().process_frame(frame, direction)
        
        # We only care about final TranscriptionFrames going downstream to the LLM aggregator
        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            text = getattr(frame, "text", "").strip()
            if not text:
                await self.push_frame(frame, direction)
                return

            try:
                # 1. Detect Language natively from STT audio
                stt_lang = getattr(frame, "language", None)
                word_count = len(text.split())

                # 2. Only update global state if we are confident (>= 3 words)
                if stt_lang and word_count >= 3:
                    if self.customer_profile.get("detected_language") != stt_lang:
                        self.customer_profile["detected_language"] = stt_lang
                        logger.info(f"Language dynamically switched to {stt_lang}. Updating TTS settings.")
                        await self.push_frame(
                            TTSUpdateSettingsFrame(
                                delta=SarvamTTSService.Settings(
                                    language=stt_lang
                                )
                            ),
                            direction
                        )

                # 3. Use active language for this turn
                detected_lang = self.customer_profile.get("detected_language", "en-IN")

                # 4. Do not translate. Pass native text directly to LLM to save 800ms.
                # LLM handles translation implicitly.
                
                await self.push_frame(frame, direction)
                
            except Exception as e:
                logger.error(f"Error in InputTranslationProcessor: {e}")
                await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)
