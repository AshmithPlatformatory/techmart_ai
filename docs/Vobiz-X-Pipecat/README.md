# Vobiz + Pipecat AI Voice Agent

AI-powered outbound calling system built with [Vobiz](https://vobiz.ai) telephony API and [Pipecat](https://pipecat.ai) voice agent framework.

## Features

- 🤖 **AI Voice Conversations** - Natural conversations powered by OpenAI GPT + TTS/STT
- 📞 **Outbound Calling** - Trigger calls via REST API from anywhere
- 🎙️ **Automatic Recording** - All conversations automatically recorded and saved
- 🔄 **Real-time Streaming** - Bidirectional audio via WebSockets
- 🚀 **Simple Setup** - Just Python, no Docker required

## How It Works

```
curl POST → Vobiz API → Call initiated → Call answered →
Vobiz requests XML → Server returns WebSocket URL →
Audio streams → Pipecat bot (STT → LLM → TTS) →
AI conversation + Recording saved
```

## Prerequisites

1. **Vobiz Account**
   - Sign up at [vobiz.ai](https://vobiz.ai)
   - Get your Auth ID and Auth Token
   - (Optional) Purchase a phone number for the `/start` endpoint

2. **OpenAI API Key**
   - Sign up at [platform.openai.com](https://platform.openai.com)
   - Create an API key with credits

3. **ngrok** (for local development)
   - Sign up at [ngrok.com](https://ngrok.com)
   - Download and install ngrok

4. **Python 3.10+**

## Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd outbound
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp env.example .env
```

Edit `.env` and add your credentials:
```env
OPENAI_API_KEY=sk-...
VOBIZ_AUTH_ID=MA_XXXXXXXX
VOBIZ_AUTH_TOKEN=your-token-here
PUBLIC_URL=https://your-ngrok-url.ngrok-free.app
```

## Usage

### 1. Start the Server

```bash
python server.py
```

Server runs on `http://0.0.0.0:7860`

### 2. Start ngrok (for local testing)

In a new terminal:
```bash
ngrok http 7860
```

Copy the ngrok URL (e.g., `https://abc123.ngrok-free.app`) and update `PUBLIC_URL` in `.env`

**Restart the server** to load the new PUBLIC_URL.

### 3. Make a Call

Call Vobiz API directly from anywhere:

```bash
curl -X POST https://api.vobiz.ai/api/v1/Account/YOUR_AUTH_ID/Call/ \
  -H "X-Auth-ID: YOUR_AUTH_ID" \
  -H "X-Auth-Token: YOUR_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "from": "+918011223344",
    "to": "+919148112233",
    "answer_url": "https://your-ngrok-url.ngrok-free.app/answer",
    "answer_method": "POST"
  }'
```

Replace:
- `YOUR_AUTH_ID` - Your Vobiz Auth ID
- `YOUR_AUTH_TOKEN` - Your Vobiz Auth Token
- `+918011223344` - Caller ID (from number)
- `+919148112233` - Number to call (to number)
- `your-ngrok-url.ngrok-free.app` - Your ngrok URL

### 4. Have a Conversation

1. Phone rings at the "to" number
2. Answer the call
3. Bot speaks and listens
4. Have a natural AI-powered conversation
5. Conversation is automatically recorded

## Accessing Recordings

Recordings are automatically downloaded to the `recordings/` folder as MP3 files.

You'll see in server logs:
```
[RECORDING CALLBACK] ✅ Downloaded to recordings/{recording-id}.mp3
[RECORDING CALLBACK] File size: 123456 bytes
```

### Manual Download

If auto-download fails, use curl with authentication:

```bash
curl -X GET "https://media.vobiz.ai/v1/Account/YOUR_AUTH_ID/Recording/{recording-id}.mp3" \
  -H "X-Auth-ID: YOUR_AUTH_ID" \
  -H "X-Auth-Token: YOUR_AUTH_TOKEN" \
  -o recording.mp3
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | OpenAI API key for LLM, STT, TTS |
| `VOBIZ_AUTH_ID` | ✅ Yes | Your Vobiz Auth ID |
| `VOBIZ_AUTH_TOKEN` | ✅ Yes | Your Vobiz Auth Token |
| `PUBLIC_URL` | ✅ Yes | Your ngrok URL (for local dev) |
| `VOBIZ_PHONE_NUMBER` | ❌ No | Optional, only for `/start` endpoint |
| `DEEPGRAM_API_KEY` | ❌ No | Optional, alternative STT provider |

### Customizing the Bot

Edit `bot.py` to customize the AI assistant:

**Change the bot's personality:**
```python
messages = [
    {
        "role": "system",
        "content": "You are a friendly customer service agent..."
    },
]
```

**Change TTS voice:**
```python
tts = OpenAITTSService(
    api_key=os.getenv("OPENAI_API_KEY"),
    voice="nova",  # Options: alloy, echo, fable, onyx, nova, shimmer
)
```

**Use different LLM:**
```python
llm = OpenAILLMService(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4-turbo"  # or "gpt-3.5-turbo" for cheaper
)
```

## API Endpoints

### `POST /start`
Initiate a call via your server (alternative to direct Vobiz API)

**Request:**
```json
{
  "phone_number": "+919148112233",
  "from_number": "+918011223344"  // optional
}
```

### `POST /answer`
Called by Vobiz when call is answered. Returns XML with WebSocket URL.

### `/ws`
WebSocket endpoint for bidirectional audio streaming.

### `POST /recording-finished`
Called by Vobiz when recording stops. Logs recording details.

### `POST /recording-ready`
Called by Vobiz when recording file is ready. Auto-downloads the MP3.

## Architecture

### Components

1. **server.py** - FastAPI server handling HTTP/WebSocket
2. **bot.py** - Pipecat voice agent (STT → LLM → TTS pipeline)
3. **Vobiz API** - Telephony service (call initiation, audio routing)
4. **ngrok** - Public URL tunnel (for local development)

### Call Flow

```
┌─────────────┐
│  You (curl) │
└──────┬──────┘
       │
       ↓ POST /Call/
┌─────────────────┐
│  Vobiz API      │
└──────┬──────────┘
       │
       ↓ Call initiated
┌─────────────────┐
│  Phone rings    │
└──────┬──────────┘
       │
       ↓ Call answered
┌─────────────────┐
│  Vobiz → POST   │
│  /answer        │
└──────┬──────────┘
       │
       ↓ Returns XML
┌─────────────────┐
│  <Stream>       │
│  wss://...      │
│  </Stream>      │
└──────┬──────────┘
       │
       ↓ WebSocket connect
┌─────────────────┐
│  Pipecat Bot    │
│  STT → LLM → TTS│
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│  AI Conversation│
│  + Recording    │
└─────────────────┘
```

## Troubleshooting

### Call Doesn't Connect

**Check:**
- Is `server.py` running?
- Is ngrok running?
- Does `answer_url` match your ngrok URL?
- Is phone number in E.164 format? (`+14155555678`)

**Test manually:**
```bash
curl https://your-ngrok-url.ngrok-free.app/answer
```
Should return XML.

### Bot Doesn't Speak

**Check:**
- Server logs show WebSocket connection?
- `PUBLIC_URL` set in `.env`?
- Restart server after changing `PUBLIC_URL`

### Can't Access Recordings

Vobiz recording URLs require authentication. Use curl with headers:

```bash
curl -X GET "https://media.vobiz.ai/..." \
  -H "X-Auth-ID: YOUR_AUTH_ID" \
  -H "X-Auth-Token: YOUR_AUTH_TOKEN" \
  -o recording.mp3
```

Or check `recordings/` folder for auto-downloaded files.

### ngrok URL Changes

Free ngrok URLs change on restart. When this happens:

1. Update `PUBLIC_URL` in `.env`
2. Restart `server.py`
3. Update `answer_url` in curl commands

**Solution:** Use ngrok paid plan for permanent URL.

## Vobiz XML Elements

The server returns XML that controls call behavior:

### Speak - Play TTS message
```xml
<Speak voice="WOMAN" language="en-US">
    Hello! Welcome to our service.
</Speak>
```

### Stream - Connect to WebSocket
```xml
<Stream bidirectional="true" contentType="audio/x-mulaw;rate=8000">
    wss://your-server.com/ws
</Stream>
```

### Record - Record the call
```xml
<Record
    action="https://your-server.com/recording-finished"
    recordSession="true"
    maxLength="3600"
    fileFormat="mp3"
/>
```

## Production Deployment

For production, deploy to a cloud provider instead of ngrok:

1. **Deploy server.py** to AWS/GCP/Heroku/etc.
2. **Get a domain** (e.g., `api.yourcompany.com`)
3. **Set PUBLIC_URL** to your domain
4. **Use HTTPS** (Let's Encrypt, Cloudflare, etc.)
5. **Make calls** using your production domain

## Resources

- [Vobiz Documentation](https://docs.vobiz.ai)
- [Pipecat Documentation](https://docs.pipecat.ai)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [ngrok Documentation](https://ngrok.com/docs)

## Support

For issues related to:
- **Vobiz API** - Contact Vobiz support
- **Pipecat** - Check Pipecat documentation/Discord
- **This integration** - Open an issue on GitHub

---

**Built with ❤️ using Vobiz + Pipecat**
