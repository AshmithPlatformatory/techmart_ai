"""
This module implements the LangGraph adapter for Pipecat.
It acts as a custom FrameProcessor that intercepts STT text frames,
constructs the conversational state, executes the LangGraph workflow,
and yields the generated text back to the pipeline for TTS synthesis.
"""
import uuid
import asyncio
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import TextFrame, LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame
from src.graph.workflow import invoke_graph_with_tracing

class LangGraphProcessor(FrameProcessor):
    def __init__(self, customer_profile: dict):
        super().__init__()
        self._customer_profile = customer_profile
        self._memory = []
        self._session_id = uuid.uuid4().hex

    async def process_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, TextFrame):
            user_text = frame.text.strip()
            if not user_text:
                return

            self._memory.append({"role": "user", "content": user_text})
            
            state = {
                "messages": self._memory,
                "customer_profile": self._customer_profile,
                "session_id": self._session_id,
                "active_intents": [],
                "detected_language": "en", 
                "retrieved_context": []
            }
            
            await self.push_frame(LLMFullResponseStartFrame(), direction)
            
            try:
                result = await asyncio.to_thread(invoke_graph_with_tracing, state)
                ai_message = result["messages"][-1].content
                self._memory.append({"role": "assistant", "content": ai_message})
                await self.push_frame(LLMTextFrame(ai_message), direction)
            except Exception as e:
                error_msg = "I am currently experiencing technical difficulties. Please hold on."
                await self.push_frame(LLMTextFrame(error_msg), direction)
                
            await self.push_frame(LLMFullResponseEndFrame(), direction)
        else:
            await self.push_frame(frame, direction)