"""
Main application entry point orchestrating the TechMart Voice Agent.
Mounts the Admin CMS routes and defines the FastAPI WebSocket endpoint for Exotel.
Identifies the caller via ClickHouse and spawns the Pipecat pipeline.
"""
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response
from src.admin.router import admin_router
from src.db.clickhouse_client import get_client
from src.bot.pipeline import create_pipecat_pipeline
from pipecat.pipeline.runner import PipelineRunner

app = FastAPI(title="TechMart Enterprise Voice Agent")

app.include_router(admin_router)

@app.post("/plivo-xml")
@app.get("/plivo-xml")
async def get_plivo_xml(request: Request):
    # Determine the current ngrok host to construct the websocket URL dynamically
    host = request.headers.get("host")
    protocol = "wss"
    
    xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream url="{protocol}://{host}/ws/plivo" />
</Response>"""
    return Response(content=xml_response, media_type="application/xml")

async def handle_websocket(websocket: WebSocket, client_type: str):
    await websocket.accept()
    
    start_data = await websocket.receive_text()
    msg = json.loads(start_data)
    
    stream_id = "default_stream"
    call_id = "default_call"
    
    if msg.get("event") == "start":
        stream_id = msg["start"].get("streamId", stream_id)
        call_id = msg["start"].get("callId", call_id)
        
    caller_phone = websocket.query_params.get("From", "")
    customer_profile = {}
    if caller_phone:
        client = get_client()
        query = f"SELECT * FROM customers WHERE phone = '{caller_phone}'"
        res = client.query(query)
        if res.result_rows:
            customer_profile = dict(zip(res.column_names, res.result_rows[0]))

    task, transport = create_pipecat_pipeline(websocket, stream_id, call_id, customer_profile, client_type)
    runner = PipelineRunner()

    try:
        await runner.run(task)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Call session terminated: {e}")

@app.websocket("/ws/plivo")
async def plivo_websocket_endpoint(websocket: WebSocket):
    await handle_websocket(websocket, "plivo")

@app.websocket("/ws/web")
async def web_websocket_endpoint(websocket: WebSocket):
    await handle_websocket(websocket, "web")