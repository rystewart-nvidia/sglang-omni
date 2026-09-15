# SPDX-License-Identifier: Apache-2.0
"""Opt-in real NeMo/HTTP parity. Set NEMOTRON_DIARIZATION_CHECKPOINT locally.

No checkpoint or evaluation output is downloaded or included in the repository.
Additional permitted conversational WAV fixtures can be supplied through
NEMOTRON_DIARIZATION_AUDIO_DIR. These checks establish integration parity,
not model accuracy; DER requires reference speaker annotations.
"""

import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

from sglang_omni.utils.audio import load_audio
from sglang_omni.utils.g711 import wrap_g711_as_wav

pytestmark = pytest.mark.accelerator


def _wav(waveform, sample_rate=16000):
    stream = io.BytesIO()
    sf.write(stream, waveform, sample_rate, format="WAV", subtype="FLOAT")
    return stream.getvalue()


@pytest.fixture(scope="module")
def checkpoint():
    path = os.environ.get("NEMOTRON_DIARIZATION_CHECKPOINT")
    if not path:
        pytest.skip("Set NEMOTRON_DIARIZATION_CHECKPOINT to a local gated checkpoint")
    if not Path(path).exists():
        pytest.fail(f"Checkpoint does not exist: {path}")
    if not torch.cuda.is_available():
        pytest.skip("Nemotron diarization requires CUDA")
    return path


@pytest.fixture(scope="module", params=["offline", "low_latency"])
def deployment(checkpoint, request, tmp_path_factory):
    from nemo.collections.asr.models import SortformerEncLabelModel

    from sglang_omni.models.nemotron_diarization.backend import resolve_nemo_checkpoint
    from sglang_omni.utils import find_available_port
    from tests.utils import start_server_from_cmd, stop_server

    profile = request.param
    # Construct the reference independently from the wrapper's profile table.
    direct = SortformerEncLabelModel.restore_from(
        str(resolve_nemo_checkpoint(checkpoint)), map_location="cuda:0", strict=True
    ).eval()
    settings = (
        dict(
            spkcache_len=264,
            fifo_len=40,
            chunk_len=340,
            chunk_right_context=40,
            spkcache_update_period=300,
        )
        if profile == "offline"
        else dict(
            spkcache_len=264,
            fifo_len=264,
            chunk_len=9,
            chunk_right_context=4,
            spkcache_update_period=222,
        )
    )
    for key, value in settings.items():
        setattr(direct.sortformer_modules, key, value)
    direct._check_streaming_parameters()
    directory = tmp_path_factory.mktemp(f"nemotron_{profile}")
    config = directory / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "config_cls": "NemotronDiarizationPipelineConfig",
                "model_path": checkpoint,
            }
        )
    )
    port = find_available_port()
    proc = start_server_from_cmd(
        [
            sys.executable,
            "-m",
            "sglang_omni.cli",
            "serve",
            "--config",
            str(config),
            "--diarization.factory.profile",
            profile,
            "--port",
            str(port),
        ],
        directory / "server.log",
        port,
        timeout=300,
    )
    try:
        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}", timeout=180, trust_env=False
        ) as client:
            yield direct, client
    finally:
        stop_server(proc)
        del direct
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def recordings():
    source = Path(__file__).parents[1] / "data" / "query_to_draw_8k.ulaw"
    audio = wrap_g711_as_wav(source.read_bytes(), "mulaw")
    waveform = load_audio(audio, target_sample_rate=16000)
    # 67 seconds traverses multiple offline chunks and speaker-cache updates.
    long_audio = np.tile(waveform, 14)[: 67 * 16000 + 37]
    cases = {
        "speech": audio,
        "silence": _wav(np.zeros(32000, dtype=np.float32)),
        "partial_tail": _wav(waveform[:16003]),
        "stereo_resampled": _wav(np.column_stack([waveform[::2], waveform[::2]]), 8000),
        "cache_updates": _wav(long_audio),
    }
    extra = os.environ.get("NEMOTRON_DIARIZATION_AUDIO_DIR")
    if extra:
        cases.update(
            {str(path): path.read_bytes() for path in sorted(Path(extra).glob("*.wav"))}
        )
    return cases


def _post(client, audio):
    response = client.post(
        "/v1/audio/diarizations", files={"file": ("audio.wav", audio)}
    )
    assert response.status_code == 200, response.text
    return response


def test_http_matches_direct_nemo_and_requests_are_isolated(deployment, recordings):
    direct, client = deployment
    expected = {}
    for name, audio in recordings.items():
        waveform = load_audio(audio, target_sample_rate=16000)
        with torch.inference_mode():
            lines, probabilities = direct.diarize(
                audio=[waveform],
                sample_rate=16000,
                batch_size=1,
                include_tensor_outputs=True,
                num_workers=0,
                verbose=False,
            )
        assert torch.isfinite(probabilities[0]).all()
        assert probabilities[0].shape[-1] == 8
        assert abs(probabilities[0].shape[1] - len(waveform) / 160) <= 1
        duration = len(waveform) / 16000
        segments = [
            {
                "start": float(start),
                "end": min(float(end), duration),
                "speaker": speaker,
            }
            for start, end, speaker in (line.split() for line in lines[0])
            if float(start) < duration
        ]
        segments.sort(
            key=lambda segment: (segment["start"], segment["end"], segment["speaker"])
        )
        expected[name] = {"duration": duration, "segments": segments}
        assert _post(client, audio).json() == expected[name]
    assert expected["silence"]["segments"] == []
    # Speech -> silence -> speech guards state leakage, including queued requests.
    names = ["speech", "silence", "speech"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        responses = list(pool.map(lambda name: _post(client, recordings[name]), names))
    assert len({response.headers["x-request-id"] for response in responses}) == len(
        names
    )
    for name, response in zip(names, responses):
        assert response.json() == expected[name]


@pytest.mark.parametrize(
    "audio", [b"", b"not audio", _wav(np.array([], dtype=np.float32))]
)
def test_bad_upload_returns_400_and_worker_recovers(deployment, audio):
    _, client = deployment
    response = client.post("/v1/audio/diarizations", files={"file": ("bad.wav", audio)})
    assert response.status_code == 400, response.text
    assert (
        _post(client, _wav(np.zeros(16000, dtype=np.float32))).json()["segments"] == []
    )
