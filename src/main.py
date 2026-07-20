"""
Main application entry point orchestrating the TechMart Voice Agent.
Mounts the Admin CMS routes and defines the FastAPI WebSocket endpoint for Exotel.
Identifies the caller via ClickHouse and spawns the Pipecat pipeline.
"""
import asyncio
import json
import os
os.environ["HF_HUB_OFFLINE"] = "1"  # Prevent 504 errors if HuggingFace is down
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response
from src.model_loader import get_sentence_transformer
from src.admin.router import admin_router
from src.db.clickhouse_client import get_client
from src.bot.pipeline import create_pipecat_pipeline
from src.bot.sentiment import preload_classifier
from src.graph.nodes import get_router_llm, get_synth_llm, write_call_ticket
from pipecat.workers.runner import WorkerRunner
import os
import aiohttp
from fastapi.responses import HTMLResponse

active_calls = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*50)
    print("  PRE-WARMING AI MODELS...              ")
    print("="*50)
    
    print("[1/4] Loading SentenceTransformer (Vector Search)...")
    get_sentence_transformer()
    print("      -> SentenceTransformer loaded successfully!")
    
    print("[2/4] Loading VoiceSentimentProcessor (HuggingFace)...")
    preload_classifier()
    print("      -> VoiceSentimentProcessor loaded successfully!")
    
    print("[3/4] Warming up ChatGroq LLMs...")
    get_router_llm()
    get_synth_llm()
    print("      -> ChatGroq clients authenticated successfully!")

    print("[4/4] Loading Silero VAD (Voice Activity Detection)...")
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    SileroVADAnalyzer()
    print("      -> Silero VAD loaded successfully!")

    print("[5/5] Waking up ClickHouse Database...")
    try:
        def _warm_db():
            get_client().command("SELECT 1")
        await asyncio.to_thread(_warm_db)
        print("      -> ClickHouse connected successfully!")
    except Exception as e:
        print(f"      -> Warning: ClickHouse warmup failed: {e}")

    print("="*50)
    print("  ALL MODELS READY! Starting server...  ")
    print("="*50 + "\n")
    yield

app = FastAPI(title="TechMart Enterprise Voice Agent", lifespan=lifespan)

app.include_router(admin_router)

@app.post("/vobiz-xml")
@app.get("/vobiz-xml")
async def get_vobiz_xml(request: Request):
    host = request.headers.get("host")
    protocol = "wss"

    caller_phone = ""

    call_uuid = request.query_params.get("CallUUID", "")

    # 1. Check query params first (handles GET requests)
    for key in ["From", "from", "Caller", "caller_id", "from_number", "phone_number"]:
        if request.query_params.get(key):
            caller_phone = request.query_params.get(key)
            break

    # 2. Check POST body — read it exactly once, then check all fields
    if request.method == "POST":
        try:
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                data = await request.json()
                call_uuid = call_uuid or data.get("CallUUID", "")
                if not caller_phone:
                    for key in ["From", "from", "Caller", "caller_id", "from_number", "phone_number"]:
                        val = data.get(key, "")
                        if val:
                            caller_phone = val
                            break
            else:
                # application/x-www-form-urlencoded (Vobiz default)
                form = await request.form()
                call_uuid = call_uuid or form.get("CallUUID", "")
                if not caller_phone:
                    for key in ["From", "from", "Caller", "caller_id", "from_number", "phone_number"]:
                        val = form.get(key, "")
                        if val:
                            caller_phone = val
                            break
        except Exception as e:
            print(f"[VOBIZ-XML] Error reading request body: {e}")

    print(f"[VOBIZ-XML] caller_phone extracted: {repr(caller_phone)}, CallUUID: {repr(call_uuid)}")

    ws_url = f"{protocol}://{host}/ws/vobiz"
    query_params = []
    if caller_phone:
        import urllib.parse
        encoded_phone = urllib.parse.quote(caller_phone, safe="")
        query_params.append(f"From={encoded_phone}")
    if call_uuid:
        query_params.append(f"CallUUID={call_uuid}")
        active_calls[call_uuid] = {"status": "initiated"}
        
    if query_params:
        ws_url += "?" + "&amp;".join(query_params)

    print(f"[VOBIZ-XML] WebSocket URL: {ws_url}")

    xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream bidirectional="true" contentType="audio/x-mulaw;rate=8000" keepCallAlive="true">{ws_url}</Stream>
</Response>"""
    return Response(content=xml_response, media_type="application/xml")

@app.api_route("/recording-ready", methods=["GET", "POST"])
async def recording_ready(request: Request) -> HTMLResponse:
    data = await request.form()
    recording_url = data.get("RecordUrl")
    recording_id = data.get("RecordingID")
    call_uuid = data.get("CallUUID")
    
    if recording_url and recording_id:
        os.makedirs("recordings", exist_ok=True)
        auth_id = os.environ.get("VOBIZ_AUTH_ID")
        auth_token = os.environ.get("VOBIZ_AUTH_TOKEN")
        
        if auth_id and auth_token:
            async with aiohttp.ClientSession() as session:
                async with session.get(recording_url, headers={"X-Auth-ID": auth_id, "X-Auth-Token": auth_token}) as resp:
                    if resp.status == 200:
                        audio_data = await resp.read()
                        with open(f"recordings/{recording_id}.mp3", "wb") as f:
                            f.write(audio_data)
                        print(f"[RECORDING] Downloaded {recording_id}.mp3")
    return HTMLResponse(content="<Response></Response>", media_type="application/xml")

@app.post("/transfer-to-human")
async def transfer_to_human(request: Request) -> HTMLResponse:
    agent_number = os.environ.get("TRANSFER_AGENT_NUMBER", "+911234567890")
    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Speak voice="WOMAN" language="en-US">Please hold while I transfer you to a human agent.</Speak>
    <Dial>{agent_number}</Dial>
</Response>"""
    return HTMLResponse(content=xml_content, media_type="application/xml")


async def handle_websocket(websocket: WebSocket, client_type: str, phone_number: str = ""):
    await websocket.accept()
    
    stream_id = "default_stream"
    call_id = "default_call"
    vobiz_encoding = "audio/x-mulaw"
    vobiz_sample_rate = 8000
    
    if client_type == "web":
        start_data = await websocket.receive_text()
        msg = json.loads(start_data)
        if msg.get("event") == "start":
            stream_id = msg["start"].get("streamId", stream_id)
            call_id = msg["start"].get("callId", call_id)
    elif client_type == "vobiz":
        from pipecat.serializers.vobiz import parse_vobiz_start
        try:
            parsed = await parse_vobiz_start(websocket)
            stream_id = parsed["stream_id"]
            call_id = parsed["call_id"]
            vobiz_encoding = parsed["encoding"] or vobiz_encoding
            vobiz_sample_rate = parsed["sample_rate"] or vobiz_sample_rate
        except Exception as e:
            print(f"Failed to parse Vobiz start frame: {e}")
            return
        
    if not phone_number:
        caller_phone = websocket.query_params.get("From", "")
    else:
        caller_phone = phone_number

    print(f"[WS-HANDLER] caller_phone received: {repr(caller_phone)}")
    customer_profile = {}
    if caller_phone:
        # Normalize incoming number to extract just the last 10 digits
        # (Handles Vobiz sending '0963...', '+91963...', or just '963...')
        normalized_digits = "".join(filter(str.isdigit, caller_phone))
        core_number = normalized_digits[-10:] if len(normalized_digits) >= 10 else normalized_digits
        
        def _fetch_profile(num: str):
            client = get_client()
            query = "SELECT * FROM customers WHERE phone LIKE %(phone)s"
            res = client.query(query, parameters={"phone": f"%{num}"})
            if res.result_rows:
                return dict(zip(res.column_names, res.result_rows[0]))
            return {}

        customer_profile = await asyncio.to_thread(_fetch_profile, core_number)

    query_call_uuid = websocket.query_params.get("CallUUID", call_id)
    if query_call_uuid and query_call_uuid in active_calls:
        active_calls[query_call_uuid]["websocket"] = websocket
        active_calls[query_call_uuid]["status"] = "active"
        
    worker, transport, context, graph_adapter = create_pipecat_pipeline(
        websocket, stream_id, call_id, customer_profile, client_type, vobiz_encoding, vobiz_sample_rate
    )
    runner = WorkerRunner(handle_sigint=False)

    try:
        await runner.add_workers(worker)
        await runner.run()
    except WebSocketDisconnect:
        print("[WS] Call session terminated cleanly (WebSocketDisconnect).")
    except Exception as e:
        print(f"Call session terminated: {e}")
    finally:
        # Generate the final ticket when the call completely finishes
        try:
            from langchain_core.messages import convert_to_messages
            raw_messages = [m for m in context.get_messages() if m.get("role") != "system"]
            if len(raw_messages) >= 2:
                langchain_msgs = convert_to_messages(raw_messages)
                transcript = "\n".join([f"{'User' if m.type == 'human' else 'Agent'}: {m.content}" for m in langchain_msgs])
                
                async def _save_ticket_bg():
                    try:
                        print("[CRM] Generating end-of-call ticket summary...")
                        await write_call_ticket(
                            ticket_id=graph_adapter._ticket_id,
                            session_id=graph_adapter._session_id,
                            customer_profile=customer_profile,
                            full_transcript=transcript
                        )
                        print("[CRM] Call ticket saved successfully.")
                    except Exception as e:
                        print(f"[CRM] Failed to write end-of-call ticket: {e}")
                
                # We use create_task because if the user hung up, this websocket handler 
                # is in a Cancelled state. Awaiting here would instantly raise CancelledError!
                asyncio.create_task(_save_ticket_bg())
        except Exception as e:
            print(f"[CRM] Failed to initiate end-of-call ticket: {e}")

        if query_call_uuid in active_calls:
            del active_calls[query_call_uuid]

@app.websocket("/ws/vobiz/{phone_number}")
async def vobiz_websocket_endpoint_with_phone(websocket: WebSocket, phone_number: str):
    await handle_websocket(websocket, "vobiz", phone_number)

@app.websocket("/ws/vobiz")
async def vobiz_websocket_endpoint(websocket: WebSocket):
    await handle_websocket(websocket, "vobiz", "")

@app.websocket("/ws/web")
async def web_websocket_endpoint(websocket: WebSocket):
    await handle_websocket(websocket, "web")