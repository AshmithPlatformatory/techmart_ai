import os
import aiohttp
from loguru import logger
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import Frame, LLMFullResponseEndFrame, LLMTextFrame, TranscriptionFrame

class SarvamTranslationProcessor(FrameProcessor):
    """
    Intercepts English LLM responses and translates them to the user's detected 
    language natively using Sarvam's Translate API before handing off to TTS.
    """
    def __init__(self, customer_profile: dict, context):
        super().__init__()
        self.customer_profile = customer_profile
        self.context = context
        self.api_key = os.getenv("SARVAM_API_KEY")
        self.buffer = ""
        self.english_buffer = ""
    
    async def process_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, LLMTextFrame):
            target_lang = self.customer_profile.get("detected_language", "en-IN")
            
            # If English, just pass it through instantly (no translation needed)
            if target_lang.startswith("en"):
                await self.push_frame(frame, direction)
                return
            
            self.buffer += frame.text
            self.english_buffer += frame.text
            
            # Sentence boundary detection (basic punctuation or newline)
            if any(punct in frame.text for punct in ['.', '!', '?', '\n']):
                await self._translate_and_push(self.buffer, target_lang, direction)
                self.buffer = ""
                
        elif isinstance(frame, LLMFullResponseEndFrame):
            target_lang = self.customer_profile.get("detected_language", "en-IN")
            
            if self.english_buffer.strip():
                self.context.add_message({"role": "assistant", "content": self.english_buffer.strip()})
                self.english_buffer = ""

            # Flush any remaining text in the buffer at the end of the response
            if not target_lang.startswith("en") and self.buffer.strip():
                await self._translate_and_push(self.buffer, target_lang, direction)
                self.buffer = ""
                
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _translate_and_push(self, text: str, target_lang: str, direction: FrameDirection):
        text = text.strip()
        if not text:
            return
            
        url = "https://api.sarvam.ai/translate"
        headers = {
            "Content-Type": "application/json",
            "api-subscription-key": self.api_key
        }
        payload = {
            "input": text,
            "source_language_code": "en-IN",
            "target_language_code": target_lang,
            "speaker_gender": "Female",
            "mode": "formal",
            "model": "sarvam-translate:v1"
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        translated = data.get("translated_text", text)
                        translated_frame = LLMTextFrame(translated + " ")
                        translated_frame.append_to_context = False
                        await self.push_frame(translated_frame, direction)
                    else:
                        logger.error(f"Sarvam translation failed: {await resp.text()}")
                        original_frame = LLMTextFrame(text + " ")
                        original_frame.append_to_context = False
                        await self.push_frame(original_frame, direction)
        except Exception as e:
            logger.error(f"Sarvam translation error: {e}")
            original_frame = LLMTextFrame(text + " ")
            original_frame.append_to_context = False
            await self.push_frame(original_frame, direction)


