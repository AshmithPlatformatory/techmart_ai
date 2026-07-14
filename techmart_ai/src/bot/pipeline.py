"""
Constructs the Pipecat real-time audio pipeline.
Configures the Exotel WebSocket transport and integrates Sarvam AI for STT/TTS.
Links the custom LangGraph adapter into the processing chain.
"""
import os
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.transcriptions.language import Language
from src.bot.adapter import LangGraphProcessor
from fastapi import WebSocket
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame
import base64
import json
from src.bot.sentiment import VoiceSentimentProcessor


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


def create_pipecat_pipeline(websocket: WebSocket, stream_id: str, call_id: str, customer_profile: dict, client_type: str = "plivo"):
    sarvam_api_key = os.environ.get("SARVAM_API_KEY")

    if client_type == "plivo":
        serializer = PlivoFrameSerializer(stream_id=stream_id)
    else:
        serializer = WebFrameSerializer()

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_out_enabled=True,
            add_wav_header=False,
            audio_in_enabled=True,
            audio_out_sample_rate=16000,
            audio_in_sample_rate=16000,
            serializer=serializer
        )
    )

    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer())

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
        sample_rate=16000,
        settings=SarvamTTSService.Settings(
            language=Language.HI_IN,
            voice="anushka",
            pace=1.0
        )
    )

    graph_adapter = LangGraphProcessor(customer_profile=customer_profile)
    sentiment_processor = VoiceSentimentProcessor()

    pipeline_elements = [transport.input()]

    pipeline_elements.extend([
        vad,
        sentiment_processor,
        stt,
        graph_adapter,
        tts,
        transport.output()
    ])

    pipeline = Pipeline(pipeline_elements)

    task = PipelineTask(pipeline)
    return task, transport