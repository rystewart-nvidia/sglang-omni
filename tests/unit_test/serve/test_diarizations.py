# SPDX-License-Identifier: Apache-2.0
"""Upload and live diarization endpoints through the real client."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from sglang_omni.client import Client
from sglang_omni.serve import create_app
from sglang_omni.serve.diarization_ws import DiarizationSession

ARCHITECTURE = "Nemotron3DiarizationForAudioFrameClassification"


class Coordinator:
    def __init__(self, result=None, stages=("diarization",)):
        self.result = result
        self.stages = list(stages)
        self.requests = []
        self.sessions = {}
        self.aborted = []

    def health(self):
        return {"stages": self.stages, "entry_stage": "diarization"}

    async def submit(self, request_id, request):
        self.requests.append((request_id, request))
        if self.result is not None:
            return self.result
        session, operation = request.inputs["session_id"], request.inputs["operation"]
        duration = 0
        if operation == "open":
            self.sessions[session] = 0
        elif operation == "reset":
            self.sessions.pop(session, None)
            self.sessions[request.inputs["next_session_id"]] = 0
        elif operation == "append":
            self.sessions[session] += len(request.inputs["pcm"]) / 32000
            duration = self.sessions[session]
        elif operation == "finish":
            duration = self.sessions.pop(session)
        else:
            self.sessions.pop(session, None)
        return {"diarization": {"duration": duration, "segments": []}}

    async def abort(self, request_id):
        self.aborted.append(request_id)

    def operations(self):
        return [request.inputs["operation"] for _, request in self.requests]


def connect(coordinator, architecture=ARCHITECTURE):
    app = create_app(
        Client(coordinator), model_name="diarizer", architectures=[architecture]
    )
    return TestClient(app)


def upload(coordinator, architecture=ARCHITECTURE, **data):
    return connect(coordinator, architecture).post(
        "/v1/audio/diarizations", data=data, files={"file": ("test.wav", b"RIFF")}
    )


def test_upload_returns_overlapping_intervals_with_request_identity():
    result = {
        "duration": 2.0,
        "segments": [
            {"start": 0.0, "end": 1.5, "speaker": "speaker_0"},
            {"start": 1.0, "end": 2.0, "speaker": "speaker_7"},
        ],
    }
    coordinator = Coordinator({"diarization": result})
    response = upload(coordinator)
    assert response.status_code == 200
    assert response.json() == result
    [(request_id, request)] = coordinator.requests
    assert response.headers["x-request-id"] == request_id
    assert request.inputs["audio_bytes"] == b"RIFF"
    assert request.metadata["task"] == "diarization"


@pytest.mark.parametrize(
    "data, status",
    [
        ({"stream": "true"}, 400),
        ({"response_format": "text"}, 400),
        ({"language": "en"}, 400),
        ({"model": "unknown"}, 404),
    ],
)
def test_unsupported_controls_or_models_are_rejected_before_dispatch(data, status):
    coordinator = Coordinator({})
    assert upload(coordinator, **data).status_code == status
    assert coordinator.requests == []


def test_missing_backend_result_is_an_error():
    assert upload(Coordinator({"text": ""})).status_code == 500


@pytest.mark.parametrize(
    "architecture, stages",
    [
        ("Qwen3ASRForConditionalGeneration", ["diarization"]),
        (ARCHITECTURE, ["diarization_0", "diarization_1"]),
    ],
    ids=["other-architecture", "process-replicas"],
)
def test_unsupported_servers_reject_diarization_without_dispatch(architecture, stages):
    coordinator = Coordinator(stages=stages)
    with connect(coordinator, architecture) as client:
        with client.websocket_connect("/v1/audio/diarizations/stream") as ws:
            assert ws.receive_json()["type"] == "error"
            assert ws.receive()["code"] == 1008
    if architecture != ARCHITECTURE:
        assert upload(coordinator, architecture).status_code == 400
    assert coordinator.requests == []


def test_live_updates_precede_eof_and_reset_replaces_the_session():
    coordinator = Coordinator()
    with connect(coordinator) as client:
        with client.websocket_connect("/v1/audio/diarizations/stream") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "session.ready"
            ws.send_bytes(b"\0\0" * 8000)
            assert ws.receive_json() == {
                "type": "diarization.update",
                "start": 0,
                "end": 0.5,
                "segments": [],
            }
            assert ws.receive_json()["type"] == "audio.ack"
            ws.send_json({"type": "session.reset"})
            reset = ws.receive_json()
            assert reset["type"] == "session.ready"
            assert reset["session_id"] != ready["session_id"]
            ws.send_bytes(b"\0\0" * 160)
            assert ws.receive_json()["start"] == 0
            assert ws.receive_json()["type"] == "audio.ack"
            ws.send_json({"type": "audio.end"})
            assert ws.receive_json() == {"type": "diarization.done", "duration": 0.01}
            assert ws.receive()["type"] == "websocket.close"
    assert coordinator.sessions == {}
    assert coordinator.operations() == [
        "open",
        "append",
        "reset",
        "append",
        "finish",
        "close",
    ]


@pytest.mark.parametrize("invalid", [b"x", b"\0" * 32002, {"type": "unknown"}])
def test_invalid_messages_close_the_session_without_dispatching_audio(invalid):
    coordinator = Coordinator()
    with connect(coordinator) as client:
        with client.websocket_connect("/v1/audio/diarizations/stream") as ws:
            assert ws.receive_json()["type"] == "session.ready"
            if isinstance(invalid, bytes):
                ws.send_bytes(invalid)
            else:
                ws.send_json(invalid)
            assert ws.receive_json()["type"] == "error"
            assert ws.receive()["type"] == "websocket.close"
    assert coordinator.sessions == {}
    assert coordinator.operations() == ["open", "close"]


def test_canceled_admission_settles_before_session_can_be_closed():
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        class SlowOpen(Coordinator):
            async def submit(self, request_id, request):
                if request.inputs["operation"] == "open":
                    started.set()
                    await release.wait()
                return await super().submit(request_id, request)

        coordinator = SlowOpen()
        session = DiarizationSession(None, client=Client(coordinator), model_name="m")
        pending = asyncio.create_task(session.request("open"))
        await started.wait()
        pending.cancel()
        await asyncio.sleep(0)
        assert not pending.done()  # Admission must settle before teardown continues.
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await session.request("close")
        assert coordinator.operations() == ["open", "close"]
        assert coordinator.sessions == {}

    asyncio.run(scenario())


def test_queue_overflow_aborts_pending_work_and_releases_session():
    async def scenario():
        ready, started = asyncio.Event(), asyncio.Event()
        outputs = []

        class SlowAppend(Coordinator):
            async def submit(self, request_id, request):
                if request.inputs["operation"] == "append":
                    started.set()
                    await asyncio.Event().wait()
                return await super().submit(request_id, request)

        class Socket:
            count = 0

            async def receive(self):
                await ready.wait()
                if self.count:
                    await started.wait()
                self.count += 1
                return {"type": "websocket.receive", "bytes": b"\0\0" * 160}

            async def send_json(self, event):
                outputs.append(event)
                if event["type"] == "session.ready":
                    ready.set()

            async def close(self):
                pass

        coordinator = SlowAppend()
        session = DiarizationSession(
            Socket(), client=Client(coordinator), model_name="m"
        )
        await asyncio.wait_for(session.run(), 2)
        assert any(
            event["type"] == "error" and "queue is full" in event["message"]
            for event in outputs
        )
        assert len(coordinator.aborted) == 1
        assert coordinator.sessions == {}
        assert session.messages.qsize() <= 8

    asyncio.run(scenario())
