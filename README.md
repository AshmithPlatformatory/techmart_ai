# TechMart Enterprise Voice Agent

## Overview

The TechMart Enterprise Voice Agent is a real-time, stateless cognitive voice engine built to handle enterprise-level customer interactions. It bridges Exotel audio streams with a multi-agent LangGraph core, utilizing Pipecat for low-latency transport and Sarvam for Speech-to-Text (STT) and Text-to-Speech (TTS) capabilities.

The system features dynamic intent routing, retrieval-augmented generation (RAG) using local embeddings (SentenceTransformers) against an AWS-hosted ClickHouse database, and real-time observability via Langfuse. 

### Key Differentiators

* **Low-Latency Architecture**: Employs an event-driven Pipecat WebSocket transport to handle 8kHz linear PCM audio streams natively.
* **Intelligent Routing**: Utilizes a fast LLM (e.g., Llama 3.1 8B via Groq) as an Intent Router to dispatch specific cognitive tasks (e.g., fetching product specs, order history, or FAQs) to parallel LangGraph worker nodes.
* **Robust Telemetry**: Granular request tracing and token usage monitoring via Langfuse integration at the application root.
* **Dialect & Tone Awareness**: Sarvam STT integration detects caller dialect and manages barge-ins accurately through Server-Side Voice Activity Detection (VAD).

## System Architecture

* **Framework**: FastAPI (WebSocket and REST Admin API)
* **Voice Transport**: Pipecat-AI, Exotel
* **Cognitive Engine**: LangGraph, Langchain (Groq LLM)
* **Speech Services**: Sarvam STT/TTS
* **Vector Database & Data Warehouse**: ClickHouse
* **Local Embedding**: Sentence-Transformers (all-MiniLM-L6-v2)
* **Observability**: Langfuse

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

Run the FastAPI application using Uvicorn. This will expose both the Admin CMS UI and the Exotel WebSocket endpoint on port 8000.

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

2. **Expose Localhost via Ngrok**

To allow Exotel to route calls to your local WebSocket endpoint, you need to expose your server to the internet using Ngrok.

```bash
# In a separate terminal
ngrok http 8000
```
Note the `wss://` URL provided by Ngrok (e.g., `wss://<ngrok-id>.ngrok-free.app/ws/exotel`) and configure it within your Exotel App Bazaar voicebot flow.

### Project Structure

* `src/main.py`: Main application entry point orchestrating the FastAPI routes and WebSocket endpoint.
* `src/admin/`: Admin CMS for updating database vectors (Product Catalog, FAQs, TOS).
* `src/bot/`: Real-time audio pipeline definitions using Pipecat.
* `src/core/`: Application core utilities, configuration, and Langfuse telemetry setup.
* `src/db/`: ClickHouse schemas, client configuration, and embedding management.
* `src/graph/`: Multi-agent LangGraph workflow definitions, Intent Router, and synthesizers.

## Testing

Ensure your system handles barge-ins and dialect switching correctly by executing test calls against the Ngrok proxy URL. Monitor the Langfuse dashboard to analyze parallel execution times for worker nodes and token costs.
