# SPDX-License-Identifier: Apache-2.0
"""Guard decoding, device placement, and admission before optional inference."""

import io
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from sglang_omni.client import Client, DiarizationResult, GenerateRequest
from sglang_omni.models.nemotron_diarization import stages
from sglang_omni.models.nemotron_diarization.backend import NemotronDiarizer
from sglang_omni.proto import StagePayload


@pytest.fixture
def executor(monkeypatch):
    import sglang_omni.platforms as platforms

    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(device_type="cuda")
    )
    calls = []

    class RecordingDiarizer:
        def __init__(self, model_path, *, device, profile):
            # A hard-coded cuda:0 would put every stage on the wrong GPU.
            assert device == torch.device("cuda:3")

        def diarize(self, waveform):
            calls.append(waveform)
            return DiarizationResult(duration=len(waveform) / 16000, segments=[])

    monkeypatch.setattr(stages, "NemotronDiarizer", RecordingDiarizer)
    scheduler = stages.create_diarization_executor("unused", device="cuda", gpu_id=3)
    assert scheduler._max_concurrency == 1  # Serialize inference to bound GPU memory.
    return scheduler, calls


def payload(audio, *, task="diarization"):
    request = Client._build_omni_request(
        GenerateRequest(
            prompt={"audio_bytes": audio}, stream=False, metadata={"task": task}
        )
    )
    return StagePayload(request_id="recording", request=request, data={})


def wav(samples, rate):
    stream = io.BytesIO()
    sf.write(stream, samples, rate, format="WAV", subtype="FLOAT")
    return stream.getvalue()


def test_stereo_resampling_preserves_duration_and_request_identity(executor):
    scheduler, calls = executor
    audio = wav(np.column_stack([np.ones(8000), -np.ones(8000)]), 8000)
    result = scheduler._fn(payload(audio))
    assert calls[0].shape == (16000,)
    np.testing.assert_array_equal(calls[0], np.zeros(16000))
    assert result.request_id == "recording"
    assert result.data == {"diarization": {"duration": 1.0, "segments": []}}


@pytest.mark.parametrize(
    "audio",
    [b"garbage", b"", wav(np.array([]), 16000), wav(np.full(1600, np.nan), 16000)],
)
def test_invalid_audio_never_reaches_inference(executor, audio):
    scheduler, calls = executor
    with pytest.raises(ValueError, match="could not decode the uploaded audio"):
        scheduler._fn(payload(audio))
    assert not calls


def test_transcription_request_never_runs_diarization(executor):
    scheduler, calls = executor
    with pytest.raises(ValueError, match="/v1/audio/diarizations"):
        scheduler._fn(payload(b"unused", task="asr"))
    assert not calls


@pytest.mark.parametrize(
    "params",
    [
        {"stream": True},
        {"temperature": 0.5},
        {"max_new_tokens": 20},
        {"num_speakers": 9},
    ],
)
def test_low_level_client_cannot_silently_apply_generation_controls(executor, params):
    scheduler, calls = executor
    request = payload(wav(np.zeros(16000), 16000))
    request.request.params.update(params)
    with pytest.raises(ValueError, match="Unsupported diarization controls"):
        scheduler._fn(request)
    assert not calls


def test_native_loader_does_not_require_nemo(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "nemo", None)
    with pytest.raises(FileNotFoundError, match="Missing checkpoint"):
        NemotronDiarizer(str(tmp_path / "missing.nemo"), device=torch.device("cuda:0"))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"device": "cpu"}, "CUDA device"),
        ({"device": "cuda:0", "profile": "live"}, "Unknown diarization profile"),
    ],
)
def test_unsupported_deployment_fails_before_checkpoint_download(kwargs, message):
    with pytest.raises(ValueError, match=message):
        NemotronDiarizer("missing/repository", **kwargs)
