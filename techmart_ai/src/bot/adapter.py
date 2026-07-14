"""
This module implements the LangGraph adapter for Pipecat.
It acts as a custom FrameProcessor that intercepts STT text frames,
constructs the conversational state, executes the LangGraph workflow,
and yields the generated text back to the pipeline for TTS synthesis.
"""
import uuid
import math
import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import (
    TextFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    LLMFullResponseEndFrame,
    EndFrame,
    OutputAudioRawFrame,
    InterruptionFrame,
    CancelFrame,
)
from src.bot.sentiment import VoiceSentimentFrame
from src.graph.workflow import stream_graph_with_tracing


class LangGraphProcessor(FrameProcessor):
    def __init__(self, customer_profile: dict):
        super().__init__()
        self._customer_profile = customer_profile
        self._memory = []
        self._session_id = uuid.uuid4().hex
        self._handoff_status = "None"
        self._user_emotion = "neutral"
        self._graph_task = None

    async def process_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        await super().process_frame(frame, direction)

        if isinstance(frame, (InterruptionFrame, CancelFrame)):
            if self._graph_task and not self._graph_task.done():
                self._graph_task.cancel()
            await self.push_frame(frame, direction)

        elif isinstance(frame, VoiceSentimentFrame):
            self._user_emotion = frame.emotion
            await self.push_frame(frame, direction)

        elif isinstance(frame, TextFrame):
            user_text = frame.text.strip()
            if not user_text:
                return

            self._memory.append(HumanMessage(content=user_text))

            state = {
                "messages": list(self._memory),
                "customer_profile": self._customer_profile,
                "session_id": self._session_id,
                "active_intents": [],
                "detected_language": "en",
                "retrieved_context": [],
                "handoff_status": self._handoff_status,
                "user_emotion": self._user_emotion,
            }

            self._graph_task = asyncio.create_task(self._run_graph(state, direction))

        else:
            await self.push_frame(frame, direction)

    async def _run_graph(self, state: dict, direction: FrameDirection):
        try:
            await self.push_frame(LLMFullResponseStartFrame(), direction)

            full_ai_message = ""
            final_state = state
            async for stream_type, data in stream_graph_with_tracing(state):
                if stream_type == "messages":
                    chunk, metadata = data
                    if metadata.get("langgraph_node") == "synthesizer":
                        content = chunk.content
                        if content:
                            full_ai_message += content
                            await self.push_frame(LLMTextFrame(content), direction)
                elif stream_type == "values":
                    final_state = data

            self._handoff_status = final_state.get("handoff_status", "None")
            self._memory.append(AIMessage(content=full_ai_message))

            if self._handoff_status == "Accepted":
                sample_rate = 16000
                duration = 0.5
                frequency = 800
                audio_data = bytearray()
                for i in range(int(sample_rate * duration)):
                    sample = int(16000 * math.sin(2 * math.pi * frequency * i / sample_rate))
                    audio_data.extend(sample.to_bytes(2, byteorder="little", signed=True))

                await self.push_frame(
                    OutputAudioRawFrame(
                        audio=bytes(audio_data),
                        sample_rate=sample_rate,
                        num_channels=1,
                    ),
                    direction,
                )
                await self.push_frame(EndFrame(), direction)
                return

            await self.push_frame(LLMFullResponseEndFrame(), direction)

        except asyncio.CancelledError:
            # We were interrupted by the user speaking over the bot
            pass
        except Exception:
            error_msg = "I am currently experiencing technical difficulties. Please hold on."
            await self.push_frame(LLMTextFrame(error_msg), direction)
            await self.push_frame(LLMFullResponseEndFrame(), direction)