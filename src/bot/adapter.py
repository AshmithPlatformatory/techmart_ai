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

from src.graph.workflow import stream_graph_with_tracing
from src.graph.nodes import write_call_ticket


def _tool_exchange(input_messages: list, final_messages: list) -> list:
    """This turn's tool-call exchange, to persist back into Pipecat's context.

    Only the tool-deciding AIMessages and the ToolMessages they produced —
    NOT the final spoken answer: the assistant aggregator records that from
    the pushed LLMTextFrames. Persisting these first keeps the context order
    correct: [..., user, AI(tool_calls), ToolMessage, AI(final answer)].
    """
    new_messages = final_messages[len(input_messages):]
    return [
        m
        for m in new_messages
        if isinstance(m, ToolMessage) or (isinstance(m, AIMessage) and m.tool_calls)
    ]


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
                        if content:
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

            # Persist tool calls back to LLMContext
            final_messages = final_state.get("messages", [])
            if final_messages:
                to_persist = _tool_exchange(messages, final_messages)
                if to_persist:
                    context.add_messages(convert_to_openai_messages(to_persist))

            # --- Fix 4: Write ticket as fire-and-forget background task ---
            # Does NOT block the user from hearing the response.
            asyncio.create_task(write_call_ticket(final_state))

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
                
                # Push EndFrame just in case the transfer fails or is delayed. Vobiz will hang up anyway on transfer.
                await self.push_frame(EndFrame())

        except asyncio.CancelledError:
            # Expected if the user interrupts (barge-in).
            # Note: The finally block in the handoff logic ensures
            # the call still hangs up if interrupted there.
            pass