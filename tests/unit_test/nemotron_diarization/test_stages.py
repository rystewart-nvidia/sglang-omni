# SPDX-License-Identifier: Apache-2.0
"""Guard decoding, device placement, admission, and GPU settlement."""

import io
import threading
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

import sglang_omni.platforms as platforms
from sglang_omni.admission import QueueFullError
from sglang_omni.client import Client, GenerateRequest
from sglang_omni.models.nemotron_diarization import stages
from sglang_omni.models.nemotron_diarization.backend import load_diarization_model
from sglang_omni.models.nemotron_diarization.model import NemotronDiarizationModel
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.step_scheduler import StepResult, StepScheduler


@pytest.fixture(autouse=True)
def cuda_platform(monkeypatch):
    monkeypatch.setattr(
        platforms,
        "current_platform",
        SimpleNamespace(device_type="cuda", supports_torchaudio_resample=lambda: True),
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(synchronize=lambda: None),
    )


class RecordingModel:
    profile = (8, 8, 340, 40, 3)
    iter_chunks = NemotronDiarizationModel.iter_chunks

    def __init__(self):
        self.calls = []

    def preprocessor(self, signal):
        self.calls.append(signal[0].numpy())
        count = signal.shape[1] // 160
        return torch.zeros(1, 128, count), count

    def forward_chunk(self, features, state, right):
        return torch.zeros(1, features.shape[2] - right, 8)


@contextmanager
def running(worker, build=None, *, max_running_requests=16, max_queued_requests=64):
    scheduler = StepScheduler(
        build or (lambda payload: stages.DiarizationTask(worker, payload)),
        max_workers=1,
        max_running_requests=max_running_requests,
        max_queued_requests=max_queued_requests,
        shutdown_callback=worker.live_sessions.close,
    )
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    try:
        yield scheduler
    finally:
        scheduler.stop()
        thread.join(timeout=5)
        assert not thread.is_alive()


def run_payload(worker, request):
    with running(worker) as scheduler:
        scheduler.inbox.put(IncomingMessage(request.request_id, "new_request", request))
        result = scheduler.outbox.get(timeout=5)
    if result.type == "error":
        raise result.data
    return result.data


def payload(prompt, *, task="diarization", request_id="recording", **params):
    request = Client.build_omni_request(
        GenerateRequest(prompt=prompt, stream=False, metadata={"task": task})
    )
    request.params.update(params)
    return StagePayload(request_id=request_id, request=request, data={})


def wav(samples, rate):
    stream = io.BytesIO()
    sf.write(stream, samples, rate, format="WAV", subtype="FLOAT")
    return stream.getvalue()


def test_stereo_resampling_preserves_duration_and_request_identity():
    model = RecordingModel()
    worker = stages.DiarizationWorker(model, torch.device("cpu"), 1, 1)
    audio = wav(np.column_stack([np.ones(8000), -np.ones(8000)]), 8000)
    result = run_payload(worker, payload({"audio_bytes": audio}))
    np.testing.assert_array_equal(model.calls[0], np.zeros(16000))
    assert result.request_id == "recording"
    assert result.data == {"diarization": {"duration": 1.0, "segments": []}}


@pytest.mark.parametrize(
    "prompt, task, params, message",
    [
        ({"audio_bytes": b"garbage"}, "diarization", {}, "could not decode"),
        ({"audio_bytes": wav(np.array([]), 16000)}, "diarization", {}, "decode"),
        (
            {"audio_bytes": wav(np.full(160, np.nan), 16000)},
            "diarization",
            {},
            "decode",
        ),
        ({"audio_bytes": b"unused"}, "asr", {}, "/v1/audio/diarizations"),
        ({"audio_bytes": b"unused"}, "diarization", {"num_speakers": 9}, "controls"),
    ],
)
def test_invalid_requests_never_reach_inference(prompt, task, params, message):
    model = RecordingModel()
    worker = stages.DiarizationWorker(model, torch.device("cpu"), 1, 1)
    with pytest.raises(ValueError, match=message):
        run_payload(worker, payload(prompt, task=task, **params))
    assert not model.calls


def test_unsupported_deployments_fail_before_loading(monkeypatch):
    devices = []

    def load(model_path, *, device, profile):
        devices.append(device)
        raise LookupError("loaded")

    for device, profile, message in [
        (torch.device("cpu"), "offline", "CUDA device"),
        (torch.device("cuda:0"), "live", "Unknown diarization profile"),
    ]:
        with pytest.raises(ValueError, match=message):
            load_diarization_model("missing/repository", device=device, profile=profile)
    monkeypatch.setattr(stages, "load_diarization_model", load)
    for concurrency in [0, True]:
        with pytest.raises(ValueError, match="max_concurrency must be a positive"):
            stages.create_diarization_executor("unused", max_concurrency=concurrency)
    assert not devices
    with pytest.raises(LookupError):
        stages.create_diarization_executor("unused", device="cuda", gpu_id=3)
    # A hard-coded cuda:0 would put every stage on the wrong GPU.
    assert devices == [torch.device("cuda:3")]


def test_failed_inference_waits_for_gpu_before_propagating_error(monkeypatch):
    launched, waiting, finished = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )

    class Stream:
        def wait_event(self, event):
            pass

        def synchronize(self):
            assert launched.is_set()
            waiting.set()
            assert finished.wait(timeout=5)

    class FailingModel(RecordingModel):
        def preprocessor(self, signal):
            launched.set()
            raise RuntimeError("inference failed after launching GPU work")

    monkeypatch.setattr(
        torch.cuda, "Event", lambda: SimpleNamespace(record=lambda stream: None)
    )
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: Stream())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    worker = stages.DiarizationWorker(FailingModel(), torch.device("cpu"), 2, 1)
    errors = []

    def run():
        try:
            run_payload(worker, payload({"audio_bytes": wav(np.zeros(1600), 16000)}))
        except RuntimeError as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert waiting.wait(timeout=5)
        assert not errors
    finally:
        finished.set()
        thread.join(timeout=5)
    assert "inference failed" in str(errors[0])


@pytest.mark.parametrize(
    "operation, fields, pending, sessions",
    [
        ("append", {"pcm": b"\0\0" * 160}, {}, ["connected"]),
        ("reset", {"next_session_id": "replacement"}, {}, ["replacement"]),
        # A pending count stands for an aborted operation whose step still runs.
        ("close", {}, {"connected": 1}, []),
    ],
)
def test_live_operations_under_a_full_queue(operation, fields, pending, sessions):
    entered, release = threading.Event(), threading.Event()

    class BlockedTask:
        bounded_by_stage = False

        def step(self):
            entered.set()
            assert release.wait(5)
            return StepResult(done=True)

        def close(self, *, aborted):
            pass

    worker = stages.DiarizationWorker(RecordingModel(), torch.device("cpu"), 1, 1)
    worker.live_sessions.compute({"session_id": "connected", "operation": "open"})
    worker.live_sessions.pending.update(pending)
    live = payload(
        {"session_id": "connected", "operation": operation, **fields},
        task="diarization_stream",
        request_id="live",
    )
    build = lambda value: (
        stages.DiarizationTask(worker, value) if value is live else BlockedTask()
    )
    with running(
        worker, build, max_running_requests=1, max_queued_requests=1
    ) as scheduler:
        scheduler.inbox.put(IncomingMessage("active", "new_request", None))
        assert entered.wait(5)
        scheduler.inbox.put(IncomingMessage("waiting", "new_request", None))
        scheduler.inbox.put(IncomingMessage("live", "new_request", live))
        if pending:
            rejection = scheduler.outbox.get(timeout=5)
            assert rejection.request_id == "live"
            assert isinstance(rejection.data, QueueFullError)
            release.set()
        else:
            release.set()
            messages = [scheduler.outbox.get(timeout=5) for _ in range(3)]
            assert {message.request_id: message.type for message in messages}[
                "live"
            ] == "result"
        assert list(worker.live_sessions.sessions) == sessions
        assert worker.live_sessions.pending == pending
