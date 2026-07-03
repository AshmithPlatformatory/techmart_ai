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
from pipecat.serializers.exotel import ExotelFrameSerializer
from pipecat.transports.websocket.server import WebsocketServerTransport, WebsocketServerParams
from pipecat.transcriptions.language import Language
from src.bot.adapter import LangGraphProcessor

def create_pipecat_pipeline(stream_sid: str, call_sid: str, customer_profile: dict):
    sarvam_api_key = os.environ.get("SARVAM_API_KEY")

    serializer = ExotelFrameSerializer(
        stream_sid=stream_sid, 
        call_sid=call_sid
    )
    
    transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            audio_out_enabled=True, 
            add_wav_header=False
        ),
        serializer=serializer
    )

    stt = SarvamSTTService(
        api_key=sarvam_api_key,
        mode="translate",
        settings=SarvamSTTService.Settings(
            model="saaras:v3",
            vad_signals=True
        )
    )
    
    tts = SarvamTTSService(
        api_key=sarvam_api_key,
        settings=SarvamTTSService.Settings(
            language=Language.HI_IN
        )
    )

    graph_adapter = LangGraphProcessor(customer_profile=customer_profile)

    pipeline = Pipeline([
        transport.input(),
        stt,
        graph_adapter,
        tts,
        transport.output()
    ])
    
    task = PipelineTask(pipeline)
    return task, transport