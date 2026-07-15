import asyncio
import base64
import json
import uuid
import wave
import time
import os
import subprocess
import sys

# Ensure requirements are installed
def install_dependencies():
    try:
        import websockets
    except ImportError:
        print("[+] Installing missing websockets dependency...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])

install_dependencies()

import websockets
import audioop
from dotenv import load_dotenv

load_dotenv()

# Configuration
WS_URI = "ws://localhost:8000/ws/vobiz?From=9632492407"
TEST_TEXT = "I want to check my order status."
TEST_AUDIO_FILE_RAW = "test_input_raw.wav"
TEST_AUDIO_FILE_WAV = "test_input.wav"
OUTPUT_AUDIO_FILE = "test_output.wav"

async def generate_test_audio():
    print(f"[+] Generating test audio for: '{TEST_TEXT}' using Sarvam REST API...")
    
    import urllib.request
    import urllib.error
    
    sarvam_api_key = os.environ.get("SARVAM_API_KEY")
    if not sarvam_api_key:
        raise Exception("SARVAM_API_KEY not found in environment variables.")
        
    url = "https://api.sarvam.ai/text-to-speech"
    
    payload = {
        "inputs": [TEST_TEXT],
        "target_language_code": "hi-IN",
        "speaker": "anushka",
        "pitch": 0,
        "pace": 1.0,
        "loudness": 1.5,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v2"
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "api-subscription-key": sarvam_api_key,
            "Content-Type": "application/json"
        },
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            if "audios" in result and len(result["audios"]) > 0:
                audio_base64 = result["audios"][0]
                with open(TEST_AUDIO_FILE_RAW, "wb") as f:
                    f.write(base64.b64decode(audio_base64))
            else:
                raise Exception(f"Unexpected API response: {result}")
    except urllib.error.HTTPError as e:
        error_info = e.read().decode("utf-8")
        raise Exception(f"Sarvam API error {e.code}: {error_info}")
        
    print(f"[+] Downloaded Sarvam TTS to {TEST_AUDIO_FILE_RAW}")


        
    # Read the 8000Hz PCM and save a copy as TEST_AUDIO_FILE_WAV
    with wave.open(TEST_AUDIO_FILE_RAW, 'rb') as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        
    if nchannels == 2:
        frames = audioop.tomono(frames, sampwidth, 1, 1)
        
    if framerate != 8000:
        frames, _ = audioop.ratecv(frames, sampwidth, 1, framerate, 8000, None)
        
    with wave.open(TEST_AUDIO_FILE_WAV, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sampwidth)
        wf.setframerate(8000)
        wf.writeframes(frames)
        
    print(f"[+] Audio generated, resampled, and saved to {TEST_AUDIO_FILE_WAV}")

async def test_pipeline():
    await generate_test_audio()
    
    # Read the PCM audio and convert to MULAW (8000Hz)
    print(f"[+] Reading {TEST_AUDIO_FILE_WAV} and converting to MULAW...")
    with wave.open(TEST_AUDIO_FILE_WAV, 'rb') as wf:
        audio_data = wf.readframes(wf.getnframes())
        
    # Convert linear PCM to mu-law which Vobiz uses
    mulaw_data = audioop.lin2ulaw(audio_data, 2)
    
    print(f"[+] Connecting to Vobiz endpoint: {WS_URI}")
    
    try:
        async with websockets.connect(WS_URI) as ws:
            print("[+] Connected successfully!")
            
            # 1. Send Vobiz Initialization Frame (JSON)
            call_id = str(uuid.uuid4())
            stream_id = str(uuid.uuid4())
            
            start_payload = {
                "event": "start",
                "streamId": stream_id,
                "callId": call_id,
                "start": {
                    "streamId": stream_id,
                    "callId": call_id,
                    "mediaFormat": {
                        "encoding": "audio/x-mulaw",
                        "sampleRate": 8000
                    }
                }
            }
            await ws.send(json.dumps(start_payload))
            print("[+] Sent Vobiz start frame.")
            
            # Give pipeline a couple of seconds to initialize (ONNX model loads)
            await asyncio.sleep(2)
            
            # 2. Stream the audio in chunks (20ms chunks)
            chunk_size = 160  # 160 bytes of mulaw = 20ms at 8000Hz
            print(f"[+] Streaming {len(mulaw_data)} bytes of simulated speech...")
            
            for i in range(0, len(mulaw_data), chunk_size):
                chunk = mulaw_data[i:i+chunk_size]
                media_payload = {
                    "event": "media",
                    "streamId": stream_id,
                    "media": {
                        "payload": base64.b64encode(chunk).decode("utf-8")
                    }
                }
                await ws.send(json.dumps(media_payload))
                await asyncio.sleep(0.02) # simulate real-time
                
            print("[+] Finished sending audio. Waiting for VAD to detect silence...")
            
            # 3. Capture the AI response
            print("[+] Waiting for LangGraph and Sarvam TTS response...")
            received_audio_mulaw = bytearray()
            
            # Set a timeout for receiving response
            start_time = time.time()
            while time.time() - start_time < 300: # Wait up to 300 seconds (allows for first-time model downloads)
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    if isinstance(response, str):
                        data = json.loads(response)
                        if data.get("event") == "playAudio":  # VobizFrameSerializer sends 'playAudio', not 'media'
                            payload = data.get("media", {}).get("payload", "")
                            if payload:
                                decoded_chunk = base64.b64decode(payload)
                                received_audio_mulaw.extend(decoded_chunk)
                                print(".", end="", flush=True)
                        elif data.get("event") == "stop" or data.get("event") == "mark":
                            print("\n[+] Received end of stream from server.")
                            break
                except asyncio.TimeoutError:
                    if len(received_audio_mulaw) > 0:
                        # We received audio and then silence, assume complete
                        break
                except websockets.exceptions.ConnectionClosed:
                    if len(received_audio_mulaw) > 0:
                        print("\n[+] Server closed connection after responding (normal for single-turn).")
                    else:
                        print("\n[!] Server closed connection before sending any audio.")
                    break
                except Exception as e:
                    print(f"\n[!] Error while receiving: {e}")
                    break
                    
            print()
            if len(received_audio_mulaw) > 0:
                print(f"[+] PASS: Received {len(received_audio_mulaw)} bytes of MULAW audio response from pipeline!")
                # Convert back to PCM to save as standard WAV
                pcm_data = audioop.ulaw2lin(bytes(received_audio_mulaw), 2)
                with wave.open(OUTPUT_AUDIO_FILE, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(8000)
                    wf.writeframes(pcm_data)
                print(f"[+] Saved AI response to {OUTPUT_AUDIO_FILE}.")
                print("\n[✔] TEST COMPLETE: The pipeline executed correctly!")
            else:
                print("[-] FAIL: Did not receive any audio response from the pipeline.")
                print("\n[✖] TEST COMPLETE: The pipeline failed to return an audio response.")
                
    except ConnectionRefusedError:
        print(f"[-] FAIL: Could not connect to {WS_URI}. Is the FastAPI server running?")
    except Exception as e:
        print(f"[-] FAIL: Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(test_pipeline())
