import os
import aiohttp
import asyncio
from loguru import logger
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import Frame, LLMFullResponseEndFrame, LLMTextFrame, TranscriptionFrame, CancelFrame, EndFrame

TTS_SUPPORTED_LANGUAGES = ["bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN"]

class SarvamTranslationProcessor(FrameProcessor):
    """
    Intercepts English LLM responses and translates them to the user's detected 
    language natively using Sarvam's Translate API before handing off to TTS.
    Uses a Future-based Ordering Queue to prevent blocking the Pipecat event loop.
    """
    def __init__(self, customer_profile: dict, context):
        super().__init__()
        self.customer_profile = customer_profile
        self.context = context
        self.api_key = os.getenv("SARVAM_API_KEY")
        self.buffer = ""
        self.english_buffer = ""
        
        # Ordering Queue mechanism
        self._task_queue = asyncio.Queue()
        self._worker_task = asyncio.create_task(self._translation_worker())
        
    async def cleanup(self):
        await super().cleanup()
        if self._worker_task:
            await self._task_queue.put((None, None)) # Signal shutdown
            self._worker_task.cancel()

    async def _translation_worker(self):
        """Background worker that pushes translations in strict chronological order."""
        try:
            while True:
                item = await self._task_queue.get()
                task, direction = item
                if task is None:
                    self._task_queue.task_done()
                    break
                
                try:
                    translated_frame = await task
                    if translated_frame:
                        await self.push_frame(translated_frame, direction)
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error in translation worker: {e}")
                finally:
                    self._task_queue.task_done()
        except asyncio.CancelledError:
            pass

    async def process_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, (CancelFrame, EndFrame)):
            # Drain queue if pipeline is cancelled/ended
            if not self._task_queue.empty():
                try:
                    await asyncio.wait_for(self._task_queue.join(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
            await self.push_frame(frame, direction)

        elif isinstance(frame, LLMTextFrame):
            target_lang = self.customer_profile.get("detected_language", "en-IN")
            if target_lang not in TTS_SUPPORTED_LANGUAGES:
                target_lang = "en-IN"
            
            # If English, just pass it through instantly (no translation needed)
            if target_lang.startswith("en"):
                await self.push_frame(frame, direction)
                return
            
            self.buffer += frame.text
            self.english_buffer += frame.text
            
            # Sentence boundary detection
            if any(punct in frame.text for punct in ['.', '!', '?', '\n']):
                # Spawn translation request immediately in background
                translation_task = asyncio.create_task(self._fetch_translation(self.buffer, target_lang))
                # Enqueue the future to be yielded in chronological order
                await self._task_queue.put((translation_task, direction))
                self.buffer = ""
                
        elif isinstance(frame, LLMFullResponseEndFrame):
            target_lang = self.customer_profile.get("detected_language", "en-IN")
            if target_lang not in TTS_SUPPORTED_LANGUAGES:
                target_lang = "en-IN"
            
            if self.english_buffer.strip():
                self.context.add_message({"role": "assistant", "content": self.english_buffer.strip()})
                self.english_buffer = ""

            # Flush any remaining text in the buffer
            if not target_lang.startswith("en") and self.buffer.strip():
                translation_task = asyncio.create_task(self._fetch_translation(self.buffer, target_lang))
                await self._task_queue.put((translation_task, direction))
                self.buffer = ""
                
            # Wait for all queued translations to finish BEFORE pushing the EndFrame
            await self._task_queue.join()
            await self.push_frame(frame, direction)
            
        else:
            await self.push_frame(frame, direction)

    async def _fetch_translation(self, text: str, target_lang: str) -> LLMTextFrame:
        text = text.strip()
        if not text:
            return None
            
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
                        frame = LLMTextFrame(translated + " ")
                        frame.append_to_context = False
                        return frame
                    else:
                        logger.error(f"Sarvam translation failed: {await resp.text()}")
                        frame = LLMTextFrame(text + " ")
                        frame.append_to_context = False
                        return frame
        except Exception as e:
            logger.error(f"Sarvam translation error: {e}")
            frame = LLMTextFrame(text + " ")
            frame.append_to_context = False
            return frame


