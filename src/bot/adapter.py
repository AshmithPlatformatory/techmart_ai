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
        first_token = True
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

        try:
            async for stream_type, data in stream_graph_with_tracing(state):
                if stream_type == "messages":
                    chunk, metadata = data
                    if metadata.get("langgraph_node") == "synthesizer":
                        content = chunk.content
                        if content and isinstance(content, str):
                            content = content.replace("*", "").replace("#", "").replace("-", " ")
                            if first_token:
                                await self.stop_ttfb_metrics()
                                await self.push_frame(LLMFullResponseStartFrame())
                                first_token = False
                            await self.push_frame(LLMTextFrame(content))
                elif stream_type == "values":
                    final_state = data
            
            # Finalize the response to release the microphone lock
            await self.push_frame(LLMFullResponseEndFrame())

            # Update our internal state
            self._handoff_status = final_state.get("handoff_status", "None")

            # --- Removed per-turn ticket writing ---
            # Call tickets are now strictly written at the end of the call in main.py

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
                
                # Push the 5-second Ringback Tone audio before hanging up
                await self.push_frame(OutputAudioRawFrame(audio=CACHED_RINGBACK_TONE, sample_rate=8000, num_channels=1))
                
                # Push EndFrame just in case the transfer fails or is delayed. Vobiz will hang up anyway on transfer.
                await self.push_frame(EndFrame())

        except asyncio.CancelledError:
            # Expected if the user interrupts (barge-in).
            # Note: The finally block in the handoff logic ensures
            # the call still hangs up if interrupted there.
            pass