import struct
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import Frame, InputAudioRawFrame

class WavHeaderInjector(FrameProcessor):
    """
    Injects a 44-byte WAV header at the very beginning of the audio stream.
    This satisfies Sarvam's API which expects WAV container formatting for its
    audio/wav endpoint, preventing audio misparsing and language hallucinations.
    """
    def __init__(self):
        super().__init__()
        self.header_sent = False

    async def process_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            if not self.header_sent:
                # 16-bit PCM (2 bytes per sample)
                byte_rate = frame.sample_rate * frame.num_channels * 2
                block_align = frame.num_channels * 2
                
                # Construct a standard 44-byte RIFF/WAVE header for streaming
                header = struct.pack(
                    '<4sI4s4sIHHIIHH4sI',
                    b'RIFF',
                    0xFFFFFFFF, # RIFF chunk size (infinite)
                    b'WAVE',
                    b'fmt ',
                    16,         # fmt chunk size
                    1,          # format: PCM
                    frame.num_channels,
                    frame.sample_rate,
                    byte_rate,
                    block_align,
                    16,         # bits per sample
                    b'data',
                    0xFFFFFFFF  # data chunk size (infinite)
                )
                
                # Prepend the 44-byte header to the very first audio chunk
                frame.audio = header + frame.audio
                self.header_sent = True
            
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)
