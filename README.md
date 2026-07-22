# TechMart Enterprise Voice Agent

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103.1-green.svg)](https://fastapi.tiangolo.com/)
[![Pipecat](https://img.shields.io/badge/Pipecat-0.0.108-orange.svg)](https://pipecat.ai/)

## Overview

The TechMart Enterprise Voice Agent is a real-time, stateless cognitive voice engine built to handle enterprise-level customer interactions. It bridges **Vobiz.ai** audio streams with a multi-agent **LangGraph** core, utilizing **Pipecat** for low-latency transport and **Sarvam AI** for Speech-to-Text (STT) and Text-to-Speech (TTS) capabilities.

The system features dynamic intent routing, retrieval-augmented generation (RAG) using local embeddings (`SentenceTransformers`) against an AWS-hosted **ClickHouse** database, and real-time observability via **Langfuse**. 

### Key Differentiators

* **Low-Latency Architecture**: Employs an event-driven Pipecat WebSocket transport to handle 8kHz linear PCM audio streams natively.
* **Dual-LLM Routing Engine**: Utilizes a heavy reasoning LLM (`Llama-3.3-70b-versatile` via Groq) as an Intent Router to dispatch specific cognitive tasks (e.g., fetching product specs, order history, or FAQs) to parallel LangGraph worker nodes, while utilizing smaller, ultra-fast models (`gpt-oss-20b` class) for voice synthesis.
* **Real-time Voice Sentiment Analysis**: Features a localized HuggingFace `VoiceSentimentProcessor` that analyzes raw audio frames to detect caller emotion (e.g., angry, happy) and automatically adjusts the Synthesizer LLM's empathetic tone dynamically.
* **Seamless Multi-Lingual Translation**: Employs an `InputTranslationProcessor` to instantly translate Hindi, Marathi, or Kannada STT output into English for vector search queries, while replying natively in the caller's language.
* **Stateful Multi-Turn Memory**: Worker nodes pass fetched context as explicit `SystemMessage` objects, allowing Pipecat's `LLMContext` to perfectly persist database context across an entire conversational session without context amnesia.
* **Token Safety Truncation**: Universal text truncation ensures that no database fetch will ever exceed context window limits (400 errors), seamlessly preserving workflow stability.
* **Dialect & Tone Awareness**: Sarvam STT integration detects caller dialect using `saaras:v3` in codemix mode. Barge-ins and turn-taking are natively managed by a local Silero VAD implementation integrated directly with Pipecat for minimal latency.

## System Architecture

* **Framework**: FastAPI (WebSocket and REST Admin API)
* **Voice Transport**: Pipecat-AI, Vobiz.ai, Native WebRTC/WebSocket
* **Cognitive Engine**: LangGraph, Langchain (Groq LLM)
* **Speech Services**: Sarvam STT/TTS
* **Vector Database & Data Warehouse**: ClickHouse
* **Local Embedding**: Sentence-Transformers (all-MiniLM-L6-v2)
* **Sentiment Analysis**: HuggingFace pre-trained models
* **Observability**: Langfuse

> **Note on Vobiz Integration:** Vobiz's audio stream is structurally identical to Twilio's Media Streams WebSocket protocol. To handle Vobiz streams within Pipecat without relying on external custom scripts, this project utilizes Pipecat's native `TwilioFrameSerializer` to decode the base64 μ-law payloads seamlessly.

## Local Development Setup

Follow these instructions to clone the repository and run the application locally.

### Prerequisites

* Python 3.9+
* Windows Operating System (or Linux/macOS equivalents)
* An active ClickHouse cluster (AWS or local)
* API Keys for Groq, Sarvam, and Langfuse
* Ngrok (for exposing the local WebSocket to the public internet)

### Installation

1. **Clone the Repository**

```bash
git clone https://github.com/AshmithPlatformatory/techmart_ai.git
cd techmart_ai
```

2. **Initialize the Virtual Environment**

It is recommended to use a native Python virtual environment.

```bash
python -m venv venv
```

Activate the virtual environment:

* On Windows:
```powershell
.\venv\Scripts\activate
```

* On macOS/Linux:
```bash
source venv/bin/activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### Configuration

Ensure your `.env` file is properly configured in the root of your project directory. You will need to provide credentials for your cloud services.

```env
# Database (ClickHouse)
CLICKHOUSE_HOST=your_clickhouse_host
CLICKHOUSE_PORT=your_clickhouse_port
CLICKHOUSE_USERNAME=your_username
CLICKHOUSE_PASSWORD=your_password

# Telemetry (Langfuse)
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=your_langfuse_host

# LLM & Speech (Groq & Sarvam)
GROQ_API_KEY=your_groq_key
SARVAM_API_KEY=your_sarvam_key
```

### Running the Application

1. **Start the FastAPI Server**

Run the FastAPI application using Uvicorn. This will expose both the Admin CMS UI and the Vobiz.ai WebSocket endpoint on port 8000.

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

2. **Expose Localhost via Ngrok (For Vobiz Integration)**

To allow Vobiz.ai to route calls to your local WebSocket endpoint, you need to expose your server to the internet using Ngrok.

```bash
# In a separate terminal
.\ngrok http 8000
```
Note the `wss://` URL provided by Ngrok (e.g., `wss://<ngrok-id>.ngrok-free.app/ws/vobiz`) and configure it within your Vobiz.ai console or webhook flow.

3. **Browser Testing (Bypassing Vobiz)**

To test the voice pipeline locally without making a real phone call, you can use the built-in web client:
```bash
# In a separate terminal
python src/web_call_client/server.py
```
Open `http://localhost:8080` in your browser to interact with the voice agent via your microphone.

### Project Structure

* `src/main.py`: Main application entry point orchestrating the FastAPI routes and WebSocket endpoint.
* `src/admin/`: Admin CMS for updating database vectors (Product Catalog, FAQs, TOS).
* `src/bot/`: Real-time audio pipeline definitions using Pipecat, Sentiment Processor, and Input Translator.
* `src/core/`: Application core utilities, configuration, and Langfuse telemetry setup.
* `src/db/`: ClickHouse schemas, client configuration, and embedding management.
* `src/graph/`: Multi-agent LangGraph workflow definitions, Intent Router, and synthesizers.
* `src/web_call_client/`: Native browser client for rapid local testing over WebSockets.

## Testing

Ensure your system handles barge-ins and dialect switching correctly by executing test calls against the Ngrok proxy URL or the local web client. Monitor the Langfuse dashboard to analyze parallel execution times for worker nodes and token costs.
