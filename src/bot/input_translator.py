import os
import aiohttp
import asyncio
from loguru import logger
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import Frame, TranscriptionFrame

class InputTranslationProcessor(FrameProcessor):
    """
    Intercepts native-script TranscriptionFrames from the STT, uses Sarvam's /text-lid
    to deterministically identify the spoken language, and translates non-English text 
    to English via Sarvam's Translate API before handing off to the LLM context.
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
                    self.customer_profile["detected_language"] = stt_lang

                # 3. Use active language for this turn
                detected_lang = self.customer_profile.get("detected_language", "en-IN")

                # 2. Translate if not English
                if not detected_lang.startswith("en"):
                    english_text = await self._fetch_translation(text, detected_lang)
                    if english_text:
                        logger.info(f"InputTranslated: '{text}' ({detected_lang}) -> '{english_text}'")
                        # Mutate frame with english text
                        frame.text = english_text
                        frame.language = detected_lang  # keep a record if needed
                
                await self.push_frame(frame, direction)
                
            except Exception as e:
                logger.error(f"Error in InputTranslationProcessor: {e}")
                await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _fetch_translation(self, text: str, source_lang: str) -> str:
        """Translate the native script text into English."""
        url = "https://api.sarvam.ai/translate"
        headers = {
            "Content-Type": "application/json",
            "api-subscription-key": self.api_key
        }
        payload = {
            "input": text,
            "source_language_code": source_lang,
            "target_language_code": "en-IN",
            "speaker_gender": "Female",
            "mode": "formal",
            "model": "sarvam-translate:v1"
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("translated_text", text)
                    else:
                        logger.warning(f"Sarvam translation failed: {await resp.text()}")
        except Exception as e:
            logger.error(f"Sarvam translation error: {e}")
        return text
