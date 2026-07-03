"""
Main application entry point orchestrating the TechMart Voice Agent.
Mounts the Admin CMS routes and defines the FastAPI WebSocket endpoint for Exotel.
Identifies the caller via ClickHouse and spawns the Pipecat pipeline.
"""
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from src.admin.router import admin_router
from src.db.clickhouse_client import get_client
from src.bot.pipeline import create_pipecat_pipeline
from pipecat.pipeline.runner import PipelineRunner

app = FastAPI(title="TechMart Enterprise Voice Agent")

app.include_router(admin_router)

@app.websocket("/ws/exotel")
async def exotel_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    caller_phone = websocket.headers.get("x-exotel-caller-id") or websocket.query_params.get("From", "")
    stream_sid = websocket.headers.get("x-exotel-stream-sid", "default_stream")
    call_sid = websocket.headers.get("x-exotel-call-sid", "default_call")

    customer_profile = {}
    if caller_phone:
        client = get_client()
        query = f"SELECT * FROM customers WHERE phone = '{caller_phone}'"
        res = client.query(query)
        if res.result_rows:
            customer_profile = dict(zip(res.column_names, res.result_rows[0]))

    task, transport = create_pipecat_pipeline(stream_sid, call_sid, customer_profile)
    runner = PipelineRunner()

    try:
        await runner.run(task)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Call session terminated: {e}")