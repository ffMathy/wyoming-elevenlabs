#!/usr/bin/env python3
# Works with wyoming ≥0.3 and the current 1.6.1               <-- real helper API
import asyncio, base64, json, os, aiohttp, websockets
from wyoming.event import async_read_event, async_write_event
from wyoming.audio import AudioStart, AudioChunk, AudioStop
from wyoming.tts import Synthesize
from wyoming.asr import Transcribe, Transcript
from wyoming.server import AsyncEventHandler                  # still the base-class

RATE, WIDTH, CHANNELS = 16_000, 2, 1
AGENT_ID = os.getenv("ELEVEN_AGENT_ID")
API_KEY  = os.getenv("ELEVEN_API_KEY")
VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "Adam")
HEADERS  = {"xi-api-key": API_KEY} if API_KEY else {}

WS_URL = (
    f"wss://api.elevenlabs.io/v1/convai/conversation"
    f"?agent_id={AGENT_ID}&output_format=pcm_16000"
)

class Gateway(AsyncEventHandler):
    async def handle_event(self, event):
        # ---------- TTS ----------
        if event.is_type(Synthesize):
            text = Synthesize.from_event(event).text
            await self.write_event(AudioStart(rate=RATE, width=WIDTH, channels=CHANNELS))
            async with aiohttp.ClientSession(headers=HEADERS) as sess:
                url = (f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream")
                async with sess.post(url, json={"text": text, "output_format": "pcm_16000"}) as res:
                    async for chunk in res.content.iter_chunked(8192):
                        await self.write_event(
                            AudioChunk(rate=RATE, width=WIDTH, channels=CHANNELS, audio=chunk)
                        )
            await self.write_event(AudioStop())
            return True

        # ---------- Conversational (ASR + reply) ----------
        if event.is_type(Transcribe):
            transcript = ""
            ws = await websockets.connect(WS_URL, extra_headers=HEADERS)
            await ws.send(json.dumps({
                "type": "conversation_initiation_client_data",
                "input_format": "pcm_s16le_16",
                "voice_id": VOICE_ID,
            }))

            # microphone stream from HA
            while True:
                ev = await async_read_event(self.reader)
                if ev.is_type(AudioStop):
                    break
                if ev.is_type(AudioChunk):
                    pcm = AudioChunk.from_event(ev).audio
                    await ws.send(json.dumps({"user_audio_chunk":
                                               base64.b64encode(pcm).decode()}))

            await self.write_event(AudioStart(rate=RATE, width=WIDTH, channels=CHANNELS))
            async for msg in ws:
                data = json.loads(msg)
                if data.get("type") == "audio":
                    pcm = base64.b64decode(data["audio_event"]["audio_base_64"])
                    await self.write_event(
                        AudioChunk(rate=RATE, width=WIDTH, channels=CHANNELS, audio=pcm)
                    )
                elif data.get("type") == "transcript":
                    transcript = data["text"]
                elif data.get("type") == "end_of_conversation":
                    break
            await self.write_event(AudioStop())
            await ws.close()
            await self.write_event(Transcript(text=transcript))
            return True

        # let AsyncEventHandler keep looping
        return False

async def main():
    server = await asyncio.start_server(
        lambda r, w: Gateway(r, w).run(), "0.0.0.0", 10300
    )
    print("Gateway ready on tcp://0.0.0.0:10300")
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    if not AGENT_ID:
        raise SystemExit("Set ELEVEN_AGENT_ID first!")
    asyncio.run(main())
