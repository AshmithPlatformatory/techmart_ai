"""
This module implements the LangGraph adapter for Pipecat.
It subclasses OpenAILLMService so our LangGraph workflow becomes the LLM stage of the
voice pipeline, inheriting trace spans, TTFB metrics, and proper interruption (barge-in) handling.
"""
import uuid
import math
import asyncio
import re
import os
from typing import Any

from langchain_core.messages import (
    AIMessage,
    ToolMessage,
    SystemMessage,
    convert_to_messages,
    convert_to_openai_messages,
)
from pipecat.frames.frames import (
    OutputAudioRawFrame,
    EndFrame,
    TTSUpdateSettingsFrame,
    LLMTextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.tracing.service_decorators import traced_llm
from pipecat.transcriptions.language import Language
from pipecat.services.sarvam.tts import SarvamTTSService
from src.bot.sentiment import VoiceSentimentFrame

TTS_SUPPORTED_LANGUAGES = ["bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN"]

import struct
import math

def _generate_ringback_tone(duration_seconds=5, sample_rate=8000):
    audio_data = bytearray()
    for i in range(duration_seconds * sample_rate):
        t = i / sample_rate
        cycle = t % 3.0
        if cycle < 1.0:
            sample = int(32767 * 0.5 * (math.sin(2 * math.pi * 440 * t) + math.sin(2 * math.pi * 480 * t)))
        else:
            sample = 0
        audio_data.extend(struct.pack('<h', sample))
    return bytes(audio_data)

# Globally cached ringback tone so it is only computed once on startup
CACHED_RINGBACK_TONE = _generate_ringback_tone()

from src.graph.workflow import stream_graph_with_tracing
from src.graph.nodes import write_call_ticket



class LangGraphLLMService(OpenAILLMService):
    """Runs a compiled LangGraph graph as the Pipecat LLM stage."""

    def __init__(self, *, customer_profile: dict, call_id: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._customer_profile = customer_profile
        self._call_id = call_id
        self._session_id = uuid.uuid4().hex
        self._ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
        self._handoff_status = "None"
        self._user_emotion = "neutral"

    async def process_frame(self, frame, direction):
        if isinstance(frame, VoiceSentimentFrame):
            self._user_emotion = frame.emotion
        await super().process_frame(frame, direction)

    @traced_llm
    async def _process_context(self, context: LLMContext) -> None:
        messages = convert_to_messages(
            [m for m in context.get_messages() if m.get("role") != "system"]
        )

        # Read detected_language from customer_profile (written by LanguageInterceptor
        # after each STT TranscriptionFrame). Default to en-IN since translate mode
        # always returns English text — we only need the language for TTS output.
        detected_lang_str = self._customer_profile.get("detected_language", "en-IN")

        state = {
            "messages": messages,
            "customer_profile": self._customer_profile,
            "session_id": self._session_id,
            "active_intents": [],
            "detected_language": detected_lang_str,
            "handoff_status": self._handoff_status,
            "user_emotion": self._user_emotion,
            "ticket_id": self._ticket_id,
        }

        await self.start_ttfb_metrics()
        first_token = [True]
        final_state = state

        # --- Fix 7: Push TTS language BEFORE the response starts ---
        # This guarantees the TTS engine is configured with the correct
        # language model before it starts synthesizing the first chunk.
        detected_lang_str = self._customer_profile.get("detected_language", "en-IN")
        final_lang_str = self._customer_profile.get("detected_language", detected_lang_str)
        if final_lang_str not in TTS_SUPPORTED_LANGUAGES:
            final_lang_str = "en-IN"
        try:
            lang_enum = Language(final_lang_str)
        except ValueError:
            lang_enum = Language.EN_IN
            
        await self.push_frame(
            TTSUpdateSettingsFrame(
                delta=SarvamTTSService.Settings(language=lang_enum)
            )
        )

        filler_words = {
            "en-IN": "Just a moment...",
            "hi-IN": "एक सेकंड...",
            "kn-IN": "ಒಂದು ನಿಮಿಷ...",
            "ml-IN": "ഒരു നിമിഷം...",
            "ta-IN": "ஒரு நிமிடம்...",
            "te-IN": "ఒక నిమిషం...",
            "bn-IN": "এক মুহূর্ত...",
            "gu-IN": "એક મિનિટ...",
            "mr-IN": "एक मिनिट...",
            "pa-IN": "ਇੱਕ ਮਿੰਟ...",
            "od-IN": "ଗୋଟେ ମିନିଟ୍..."
        }
        filler = filler_words.get(final_lang_str, "Just a moment...")

        async def filler_task():
            await asyncio.sleep(0.6)
            if first_token[0]:
                first_token[0] = False
                await self.stop_ttfb_metrics()
                await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(LLMTextFrame(filler))

        filler_task_handle = asyncio.create_task(filler_task())

        # Minimal professional comment: Parses astream_events and handles tool call isolation.
        try:
            async for event in stream_graph_with_tracing(state, self._session_id):
                if event["event"] == "on_chat_model_stream":
                    if event.get("metadata", {}).get("langgraph_node") == "agent":
                        chunk = event["data"]["chunk"]
                        # Skip tool call chunks, only push text
                        if chunk.content and isinstance(chunk.content, str):
                            if not first_token_yielded:
                                first_token_yielded = True
                                first_token[0] = False
                                filler_task_handle.cancel()
                                await self.stop_ttfb_metrics()
                                await self.push_frame(LLMFullResponseStartFrame())
                            
                            cleaned_content = chunk.content.replace("*", "").replace("#", "")
                            if cleaned_content:
                                await self.push_frame(LLMTextFrame(cleaned_content))
                elif event["event"] == "on_chain_end":
                    if event.get("metadata", {}).get("langgraph_node") == "agent":
                        final_state = event["data"]["output"]
            
            # Finalize the response to release the microphone lock
            await self.push_frame(LLMFullResponseEndFrame())

            # Update our internal state
            self._handoff_status = final_state.get("handoff_status", "None")

            # Handoff logic
            if self._handoff_status == "Accepted":
                auth_id = os.environ.get("VOBIZ_AUTH_ID")
                auth_token = os.environ.get("VOBIZ_AUTH_TOKEN")
                public_url = os.environ.get("PUBLIC_URL")
                
                if auth_id and auth_token and public_url and self._call_id:
                    import aiohttp
                    vobiz_url = f"https://api.vobiz.ai/api/v1/Account/{auth_id}/Call/{self._call_id}/"
                    transfer_data = {
                        "legs": "aleg",
                        "aleg_url": f"{public_url.rstrip('/')}/transfer-to-human",
                        "aleg_method": "POST"
                    }
                    try:
                        async with aiohttp.ClientSession() as session:
                            await session.post(
                                vobiz_url,
                                headers={"X-Auth-ID": auth_id, "X-Auth-Token": auth_token},
                                json=transfer_data
                            )
                        print("[TRANSFER] Initiated transfer via Vobiz API")
                    except Exception as e:
                        print(f"[TRANSFER] Failed to initiate transfer: {e}")
                else:
                    missing = []
                    if not auth_id: missing.append("VOBIZ_AUTH_ID")
                    if not auth_token: missing.append("VOBIZ_AUTH_TOKEN")
                    if not public_url: missing.append("PUBLIC_URL")
                    if not self._call_id: missing.append("Call UUID")
                    print(f"[TRANSFER] ERROR: Handoff failed silently because these variables are missing: {', '.join(missing)}")
                
                # Push the 5-second Ringback Tone audio before hanging up
                await self.push_frame(OutputAudioRawFrame(audio=CACHED_RINGBACK_TONE, sample_rate=8000, num_channels=1))
                
                # Push EndFrame just in case the transfer fails or is delayed. Vobiz will hang up anyway on transfer.
                await self.push_frame(EndFrame())

        except asyncio.CancelledError:
            # Expected if the user interrupts (barge-in).
            # Note: The finally block in the handoff logic ensures
            # the call still hangs up if interrupted there.
            pass
        finally:
            if not filler_task_handle.done():
                filler_task_handle.cancel()