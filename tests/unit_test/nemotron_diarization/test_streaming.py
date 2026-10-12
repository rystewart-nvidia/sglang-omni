# SPDX-License-Identifier: Apache-2.0
"""Streaming acoustic boundaries, hysteresis, and session ownership."""

import numpy as np
import pytest
import torch

from sglang_omni.models.nemotron_diarization import streaming
from sglang_omni.models.nemotron_diarization.model import Preprocessor
from sglang_omni.models.nemotron_diarization.streaming import (
    LiveDiarization,
    LiveSessions,
)


class AcousticProbe:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.preprocessor.featurizer.window.copy_(torch.hann_window(400))
        self.preprocessor.featurizer.fb.fill_(0.01)
        self.chunks = []

    def forward_chunk(self, features, state, right):
        self.chunks.append(features.clone())
        return features.new_zeros(1, features.shape[2] - right, 8)


def call(manager, session, operation, pcm=b"", **fields):
    return manager.compute(
        dict(session_id=session, operation=operation, pcm=pcm, **fields)
    )


@pytest.mark.parametrize("length", [159, 16640, 50003])
def test_incremental_features_match_complete_audio_and_flush_partial_tail(length):
    model = AcousticProbe()
    audio = np.random.default_rng(17).normal(size=length).astype(np.float32)
    expected, count = model.preprocessor(torch.from_numpy(audio)[None, :])
    expected = expected[:, :, :count]
    live = LiveDiarization(model, device=torch.device("cpu"))
    outputs = []
    offset = 0
    for size in [1, 159, 161, 800, 3201] * (length // 4000 + 1):
        chunk = audio[offset : offset + size]
        if not len(chunk):
            break
        offset += len(chunk)
        outputs.append(live.probabilities(chunk))
        if model.chunks:
            assert offset >= 104 * 160 + 256  # No premature right-context padding.
        assert live.audio.size < 104 * 160 + 256 + 320
    if length > 104 * 160 + 256:
        assert sum(output.shape[1] for output in outputs) > 0  # Live, before EOF.
    outputs.append(live.probabilities(np.empty(0, np.float32), final=True))
    assert torch.cat(outputs, dim=1).shape == (1, length // 160, 8)
    assert live.audio.size == 0
    for index, chunk in enumerate(model.chunks):
        start = index * 72
        torch.testing.assert_close(
            chunk, expected[:, :, start : start + 104], rtol=0, atol=0
        )
    with pytest.raises(ValueError, match="finished"):
        live.probabilities(np.zeros(160, np.float32))


def test_equal_threshold_preserves_activity_across_updates(monkeypatch):
    live = LiveDiarization(AcousticProbe(), device=torch.device("cpu"))
    values = iter([[0.6, 0.5], [0.5, 0.4, 0.5]])

    def probabilities(audio, *, final):
        speaker = torch.tensor(next(values))
        live.frame += len(speaker)
        result = torch.zeros(1, len(speaker), 8)
        result[0, :, 0] = speaker
        result[0, :, 7] = speaker  # Overlap must survive.
        return result

    monkeypatch.setattr(live, "probabilities", probabilities)
    first, second = live.append(b""), live.append(b"")
    assert [(s.start, s.end, s.speaker) for s in first.segments] == [
        (0, 0.02, "speaker_0"),
        (0, 0.02, "speaker_7"),
    ]
    assert [(s.start, s.end, s.speaker) for s in second.segments] == [
        (0.02, 0.03, "speaker_0"),
        (0.02, 0.03, "speaker_7"),
    ]
    assert second.duration == 0.05


def test_sessions_bound_state_reject_invalid_or_concurrent_work_and_release():
    manager = LiveSessions(AcousticProbe(), torch.device("cpu"), 2)
    call(manager, "a", "open")
    call(manager, "b", "open")
    assert manager.sessions["a"].state is not manager.sessions["b"].state
    with pytest.raises(ValueError, match="limit"):
        call(manager, "c", "open")
    with pytest.raises(ValueError, match="PCM16"):
        call(manager, "a", "append", b"x")
    with manager.sessions["a"].lock:
        with pytest.raises(ValueError, match="pending"):
            call(manager, "a", "append", b"\0\0")
    call(manager, "a", "append", b"\0\0" * 111)
    assert call(manager, "a", "finish").duration == 111 / 16000
    call(manager, "b", "reset", next_session_id="c")
    assert set(manager.sessions) == {"c"}
    call(manager, "c", "close")
    with pytest.raises(ValueError, match="closed"):
        call(manager, "c", "append", b"\0\0")
    call(manager, "idle", "open")
    manager.sessions["idle"].last_used -= 61
    call(manager, "new", "open")
    assert set(manager.sessions) == {"new"}


def test_inference_failure_discards_partial_state_and_releases_capacity(monkeypatch):
    model = AcousticProbe()
    manager = LiveSessions(model, torch.device("cpu"), 1)
    call(manager, "failed", "open")
    state = manager.sessions["failed"]
    call(manager, "failed", "append", b"\0\0" * 16000)

    def fail(features, cache, right):
        cache.fifo = features.new_zeros(1, 1, 512)
        raise RuntimeError("inference failed after changing state")

    monkeypatch.setattr(model, "forward_chunk", fail)
    with pytest.raises(RuntimeError, match="after changing state"):
        call(manager, "failed", "append", b"\0\0" * 1600)
    assert manager.sessions == {}
    assert not state.lock.locked()
    call(manager, "replacement", "open")
    assert manager.sessions["replacement"].state.fifo.shape[1] == 0


def test_queued_activity_prevents_expiry_and_abort_releases_reservation(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(streaming.time, "monotonic", lambda: now[0])
    manager = LiveSessions(AcousticProbe(), torch.device("cpu"), 1)
    call(manager, "live", "open")
    now[0] = 55.0
    assert manager.reserve("live")
    now[0] = 65.0
    assert call(manager, "live", "append", b"\0\0").duration == 0.0
    manager.release("live", aborted=False)
    assert "live" in manager.sessions and not manager.pending
    manager.reserve("live")
    manager.release("live", aborted=True)
    assert not manager.sessions and not manager.pending
    call(manager, "expired", "open")
    now[0] += 61.0
    assert not manager.reserve("expired")
    with pytest.raises(ValueError, match="closed or expired"):
        call(manager, "expired", "append", b"\0\0")
