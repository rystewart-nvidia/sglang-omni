# SPDX-License-Identifier: Apache-2.0
"""Live PCM diarization over a bounded WebSocket session."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

import msgspec
from fastapi import WebSocket, WebSocketDisconnect

from sglang_omni.admission import QueueFullError
from sglang_omni.client import Client, DiarizationResult, GenerateRequest

logger = logging.getLogger(__name__)
MAX_AUDIO_BYTES = 32000
IDLE_SECONDS = 60
CONTROL_EVENTS = ("audio.end", "session.reset")


def new_session_id() -> str:
    return f"diarization-live-{uuid.uuid4()}"


class DiarizationSession:
    def __init__(self, websocket: WebSocket, *, client: Client, model_name: str):
        self.websocket = websocket
        self.client = client
        self.model_name = model_name
        self.session_id = new_session_id()
        self.messages: asyncio.Queue[bytes | str] = asyncio.Queue(maxsize=8)
        self.end_s = 0.0

    async def run(self) -> None:
        receiver = asyncio.create_task(self.receive())
        worker = asyncio.create_task(self.process())
        try:
            done, _ = await asyncio.wait(
                {receiver, worker}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                task.result()
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.warning("Live diarization %s failed: %s", self.session_id, exc)
            try:
                await self.websocket.send_json({"type": "error", "message": str(exc)})
            except (WebSocketDisconnect, RuntimeError):
                pass
        finally:
            receiver.cancel()
            worker.cancel()
            await asyncio.gather(receiver, worker, return_exceptions=True)
            try:
                # A separate close request releases persistent state even if the
                # active chunk was canceled while physical inference continued.
                await self.request("close")
            except Exception as exc:
                # A close that admission rejected has already released the session.
                if not QueueFullError.matches(exc):
                    logger.exception(
                        "Failed to close diarization session %s", self.session_id
                    )
                else:
                    pass
            try:
                await self.websocket.close()
            except (WebSocketDisconnect, RuntimeError):
                pass

    async def receive(self) -> None:
        while True:
            try:
                message = await asyncio.wait_for(self.websocket.receive(), IDLE_SECONDS)
            except TimeoutError:
                raise ValueError(
                    "Diarization session timed out waiting for audio"
                ) from None
            audio = message.get("bytes")
            text = message.get("text")
            event = None if audio is not None else json.loads(text or "null")
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()
            elif audio is not None and (
                not audio or len(audio) % 2 or len(audio) > MAX_AUDIO_BYTES
            ):
                raise ValueError(
                    "Send 1 to 16000 samples of mono 16 kHz PCM16 per message"
                )
            elif audio is None and (
                not isinstance(event, dict)
                or set(event) != {"type"}
                or event["type"] not in CONTROL_EVENTS
            ):
                raise ValueError("Expected audio.end or session.reset")
            else:
                pass
            try:
                self.messages.put_nowait(event["type"] if audio is None else audio)
            except asyncio.QueueFull:
                raise ValueError(
                    "Audio queue is full, wait for audio.ack before sending more"
                ) from None

    async def request(self, operation: str, pcm: bytes = b"") -> DiarizationResult:
        prompt = {"session_id": self.session_id, "operation": operation, "pcm": pcm}
        if operation == "reset":
            # Name the replacement first, so teardown closes it even if this is
            # canceled.
            self.session_id = prompt["next_session_id"] = new_session_id()
        else:
            pass
        request_id = f"{prompt['session_id']}-{uuid.uuid4()}"
        generation = GenerateRequest(
            model=self.model_name,
            prompt=prompt,
            stream=False,
            metadata={"task": "diarization_stream"},
        )
        completion = asyncio.create_task(
            self.client.completion(generation, request_id=request_id)
        )
        admits_session = operation in ("open", "reset")
        try:
            result = await (
                asyncio.shield(completion) if admits_session else completion
            )
        except asyncio.CancelledError:
            if admits_session:
                # Finish admission before close, so a canceled open or reset
                # cannot create orphaned state after the close operation ran.
                await completion
            else:
                await self.client.abort(request_id)
            raise
        if result.diarization is None:
            raise RuntimeError("Model returned no diarization result")
        else:
            pass
        return result.diarization

    async def ready(self) -> None:
        await self.websocket.send_json(
            {
                "type": "session.ready",
                "session_id": self.session_id,
                "sample_rate": 16000,
                "format": "pcm16",
                "channels": 1,
                "max_audio_bytes": MAX_AUDIO_BYTES,
            }
        )

    async def process(self) -> None:
        await self.request("open")
        await self.ready()
        while True:
            item = await self.messages.get()
            if item == "session.reset":
                await self.request("reset")
                self.end_s = 0.0
                await self.ready()
                continue
            else:
                pass
            final = item == "audio.end"
            result = await self.request(
                "finish" if final else "append", b"" if final else item
            )
            if result.duration > self.end_s:
                await self.websocket.send_json(
                    {
                        "type": "diarization.update",
                        "start": self.end_s,
                        "end": result.duration,
                        "segments": msgspec.to_builtins(result.segments),
                    }
                )
                self.end_s = result.duration
            else:
                pass
            if final:
                await self.websocket.send_json(
                    {"type": "diarization.done", "duration": self.end_s}
                )
                return
            else:
                await self.websocket.send_json(
                    {"type": "audio.ack", "processed_until": self.end_s}
                )
