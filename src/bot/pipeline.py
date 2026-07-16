"""
Constructs the Pipecat real-time audio pipeline.
Configures the Vobiz WebSocket transport and integrates Sarvam AI for STT/TTS.
Links the custom LangGraph adapter into the processing chain.
"""
import os
import base64
import json
from fastapi import WebSocket
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.pipeline.task import PipelineParams
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.serializers.vobiz import VobizFrameSerializer
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.transcriptions.language import Language
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter
from pipecat.turns.user_turn_strategies import TurnAnalyzerUserTurnStopStrategy, UserTurnStrategies
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
from src.bot.sentiment import VoiceSentimentProcessor
from src.bot.translator import SarvamTranslationProcessor
from src.bot.adapter import LangGraphLLMService
from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter

import re

def contains_indic_script(text: str) -> bool:
    return bool(re.search(r'[\u0900-\u0D7F]', text))

class LanguageInterceptor(FrameProcessor):
    def __init__(self, state_dict: dict):
        super().__init__()
        self.state_dict = state_dict

    async def process_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.language:
            text = getattr(frame, "text", "").strip()
            if len(text) > 0:
                self.state_dict["detected_language"] = frame.language.value
        await self.push_frame(frame, direction)



class WebFrameSerializer(FrameSerializer):
    async def serialize(self, frame):
        if isinstance(frame, OutputAudioRawFrame):
            payload = base64.b64encode(frame.audio).decode("utf-8")
            return json.dumps({"event": "media", "media": {"payload": payload}})
        return None

    async def deserialize(self, data):
        try:
            msg = json.loads(data)
            if msg.get("event") == "media":
                audio = base64.b64decode(msg["media"]["payload"])
                return InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1)
        except Exception:
            pass
        return None


def create_pipecat_pipeline(websocket: WebSocket, stream_id: str, call_id: str, customer_profile: dict, client_type: str = "vobiz", encoding: str = "audio/x-mulaw", sample_rate: int = 8000):
    sarvam_api_key = os.environ.get("SARVAM_API_KEY")
    # Force the Pipecat pipeline to process at 8000 Hz for Vobiz native
    audio_in_rate = 8000

    if client_type == "vobiz":
        serializer = VobizFrameSerializer(
            stream_id=stream_id,
            call_id=call_id,
            auth_id=os.environ.get("VOBIZ_AUTH_ID", ""),
            auth_token=os.environ.get("VOBIZ_AUTH_TOKEN", ""),
            params=VobizFrameSerializer.InputParams(
                vobiz_sample_rate=sample_rate, # The network sample rate (8000 Hz)
                encoding=encoding,
                l16_byte_order="be",
                auto_hang_up=True,
            ),
        )
    else:
        serializer = WebFrameSerializer()

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_out_enabled=True,
            add_wav_header=False,
            audio_in_enabled=True,
            serializer=serializer
        )
    )

    stt = SarvamSTTService(
        api_key=sarvam_api_key,
        mode="translate",
        settings=SarvamSTTService.Settings(
            model="saaras:v3",
            vad_signals=False
        )
    )

    tts = SarvamTTSService(
        api_key=sarvam_api_key,
        sample_rate=8000,
        text_filter=MarkdownTextFilter(),
        settings=SarvamTTSService.Settings(
            model="bulbul:v3",
            language=Language.HI_IN,
            voice="priya",
            pace=1.0
        )
    )

    # In Pipecat 1.x, VAD analyzer is bound to the LLM Context Aggregator pair
    context = LLMContext([])
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.8)),
            user_turn_strategies=UserTurnStrategies(
                start=[MinWordsUserTurnStartStrategy(min_words=2)]
            )
        ),
    )

    lang_interceptor = LanguageInterceptor(customer_profile)
    graph_adapter = LangGraphLLMService(customer_profile=customer_profile, call_id=call_id, api_key="not-used")
    sentiment_processor = VoiceSentimentProcessor()
    translator = SarvamTranslationProcessor(customer_profile, context)
    
    pipeline = Pipeline([
        transport.input(),
        sentiment_processor,
        stt,
        lang_interceptor,
        context_aggregator.user(),
        graph_adapter,
        translator,
        tts,
        transport.output(),
        context_aggregator.assistant()
    ])

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=audio_in_rate,
            audio_out_sample_rate=8000,
        )
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        from pipecat.frames.frames import TextFrame, TTSUpdateSettingsFrame, LLMMessagesAppendFrame
        from pipecat.processors.aggregators.llm_context import LLMContextMessage
        customer_name = customer_profile.get("name", "there")
        greeting = f"Hello {customer_name}, welcome to TechMart. How can I help you today?"
        
        # Delay the greeting slightly to give the Vobiz telecom network
        # time to fully establish the audio channel. Otherwise, Pipecat speaks 
        # while the phone is still ringing and the audio gets swallowed.
        import asyncio
        await asyncio.sleep(1.5)

        await worker.queue_frames([
            TTSUpdateSettingsFrame(delta=SarvamTTSService.Settings(language=Language.EN_IN)),
            LLMMessagesAppendFrame(messages=[{"role": "assistant", "content": greeting}]),
            TextFrame(greeting)
        ])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        print("[WS] Vobiz WebSocket disconnected by caller. Cancelling pipeline...")
        await worker.cancel()

    return worker, transport