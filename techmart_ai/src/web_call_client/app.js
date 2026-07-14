let ws;
let audioContext;
let mediaStream;
let sourceNode;
let processorNode;
let playTime = 0;

const startBtn = document.getElementById('startBtn');
const endBtn = document.getElementById('endBtn');
const statusText = document.getElementById('statusText');

startBtn.addEventListener('click', startCall);
endBtn.addEventListener('click', endCall);

async function startCall() {
    startBtn.style.display = 'none';
    statusText.innerText = 'Connecting...';
    
    try {
        // Connect to WebSocket (assuming backend runs on localhost:8000 for local testing)
        // Ensure your main.py is running on port 8000
        const wsUrl = 'ws://localhost:8000/ws/web?From=9632492407';
        ws = new WebSocket(wsUrl);
        
        ws.onopen = async () => {
            statusText.innerText = 'Connected. Starting audio...';
            
            // Send Plivo Start Frame
            ws.send(JSON.stringify({
                event: 'start',
                start: { streamId: 'web-stream', callId: 'web-call' }
            }));

            // Initialize Audio at 16kHz
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            playTime = audioContext.currentTime;

            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            sourceNode = audioContext.createMediaStreamSource(mediaStream);
            
            processorNode = audioContext.createScriptProcessor(4096, 1, 1);
            
            processorNode.onaudioprocess = (e) => {
                if (ws.readyState !== WebSocket.OPEN) return;
                
                const inputData = e.inputBuffer.getChannelData(0);
                
                // Convert float32 [-1.0, 1.0] to int16
                const pcm16 = new Int16Array(inputData.length);
                for (let i = 0; i < inputData.length; i++) {
                    let s = Math.max(-1, Math.min(1, inputData[i]));
                    pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                
                // Convert Int16Array to Base64
                const uint8 = new Uint8Array(pcm16.buffer);
                let binary = '';
                for (let i = 0; i < uint8.byteLength; i++) {
                    binary += String.fromCharCode(uint8[i]);
                }
                const base64Audio = btoa(binary);
                
                // Send Plivo Media Frame
                ws.send(JSON.stringify({
                    event: 'media',
                    streamId: 'web-stream',
                    media: { payload: base64Audio }
                }));
            };
            
            sourceNode.connect(processorNode);
            processorNode.connect(audioContext.destination);
            
            statusText.innerText = 'Call Active. Speak now.';
            endBtn.style.display = 'block';
        };

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.event === 'media' && msg.media && msg.media.payload) {
                playAudioChunk(msg.media.payload);
            }
        };

        ws.onclose = () => {
            endCall();
        };

    } catch (err) {
        console.error(err);
        statusText.innerText = 'Error: ' + err.message;
        startBtn.style.display = 'block';
    }
}

function playAudioChunk(base64Str) {
    if (!audioContext) return;
    
    const binary = atob(base64Str);
    const len = binary.length;
    const uint8 = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        uint8[i] = binary.charCodeAt(i);
    }
    
    const int16 = new Int16Array(uint8.buffer);
    const audioBuffer = audioContext.createBuffer(1, int16.length, 16000);
    const channelData = audioBuffer.getChannelData(0);
    
    for (let i = 0; i < int16.length; i++) {
        channelData[i] = int16[i] / 32768.0;
    }
    
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);
    
    const currTime = audioContext.currentTime;
    if (playTime < currTime) {
        playTime = currTime;
    }
    source.start(playTime);
    playTime += audioBuffer.duration;
}

function endCall() {
    if (ws) {
        ws.close();
        ws = null;
    }
    if (processorNode) {
        processorNode.disconnect();
        processorNode = null;
    }
    if (sourceNode) {
        sourceNode.disconnect();
        sourceNode = null;
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(t => t.stop());
        mediaStream = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    
    statusText.innerText = 'Call ended.';
    startBtn.style.display = 'block';
    endBtn.style.display = 'none';
}
