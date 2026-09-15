# SPDX-License-Identifier: Apache-2.0
"""Checkpoint loading and inference for the optional NeMo backend."""

from __future__ import annotations

import math
import re
import tarfile
from pathlib import Path

import numpy as np
import torch
import yaml

from sglang_omni.client.types import DiarizationResult, DiarizationSegment

CHECKPOINT_FILENAME = "Nemotron-3-Diarization-preview.nemo"
SAMPLE_RATE = 16000
NEMO_REVISION = "2c1a2f91d64566b5d391b83df42f9ab4cd810adb"
_PROFILES = {
    # Cache, FIFO, chunk, right context and update period, in 80 ms frames.
    "offline": (264, 40, 340, 40, 300),
    "low_latency": (264, 264, 9, 4, 222),
}


def resolve_nemo_checkpoint(model_path: str) -> Path:
    path = Path(model_path).expanduser()
    if path.is_file():
        if path.suffix != ".nemo":
            raise ValueError("Nemotron diarization requires a .nemo checkpoint")
        return path
    if path.is_dir():
        checkpoint = path / CHECKPOINT_FILENAME
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        return checkpoint
    if (
        path.is_absolute()
        or model_path.startswith((".", "~"))
        or path.suffix == ".nemo"
    ):
        raise FileNotFoundError(f"Missing checkpoint: {path}")

    from huggingface_hub import hf_hub_download

    repo_id, _, revision = model_path.partition("@")
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=CHECKPOINT_FILENAME,
            revision=revision or None,
        )
    )


def validate_checkpoint(path: Path) -> None:
    # Inspect configuration without instantiating any of its NeMo targets.
    with tarfile.open(path) as archive:
        configs = [
            member
            for member in archive.getmembers()
            if member.name in {"model_config.yaml", "./model_config.yaml"}
        ]
        if len(configs) != 1 or not configs[0].isfile() or configs[0].size > 1024**2:
            raise ValueError("Expected one model_config.yaml in the .nemo archive")
        with archive.extractfile(configs[0]) as handle:
            config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Invalid Nemotron diarization configuration")
    encoder = config.get("encoder", {})
    modules = config.get("sortformer_modules", {})
    preprocessor = config.get("preprocessor", {})
    if not (
        isinstance(encoder, dict)
        and isinstance(modules, dict)
        and isinstance(preprocessor, dict)
        and config.get("target")
        == "nemo.collections.asr.models.sortformer_diar_models.SortformerEncLabelModel"
        and config.get("sample_rate") == SAMPLE_RATE
        and config.get("high_resolution") is True
        and config.get("output_subsampling_factor") == 1
        and config.get("streaming_mode") is True
        and config.get("max_num_of_spks") == 8
        and preprocessor.get("window_stride") == 0.01
        and preprocessor.get("sample_rate") == SAMPLE_RATE
        and encoder.get("self_attention_model") == "rope"
        and encoder.get("n_layers") == 31
        and encoder.get("subsampling_factor") == 8
        and modules.get("num_spks") == 8
    ):
        raise ValueError(
            "Checkpoint is not the supported Nemotron 3 diarization layout"
        )


def parse_segments(lines: list[str], duration: float) -> DiarizationResult:
    """Preserve overlapping speakers and clip frame-rounded ends to the waveform."""
    segments = []
    for line in lines:
        try:
            start_text, end_text, speaker = line.split()
            start, end = float(start_text), float(end_text)
        except (ValueError, AttributeError) as exc:
            raise RuntimeError(f"Invalid NeMo diarization segment: {line!r}") from exc
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
            or re.fullmatch(r"speaker_[0-7]", speaker) is None
            or end > duration + 0.01
        ):
            raise RuntimeError(f"Invalid NeMo diarization segment: {line!r}")
        end = min(end, duration)
        if start < end:
            segments.append(DiarizationSegment(start=start, end=end, speaker=speaker))
    segments.sort(key=lambda segment: (segment.start, segment.end, segment.speaker))
    return DiarizationResult(duration=duration, segments=segments)


class NeMoDiarizer:
    def __init__(
        self, model_path: str, *, device: torch.device, profile: str = "offline"
    ):
        if profile not in _PROFILES:
            raise ValueError(
                f"Unknown diarization profile {profile!r}; use {list(_PROFILES)}"
            )
        if torch.device(device).type != "cuda":
            raise ValueError("Nemotron 3 diarization requires an NVIDIA CUDA device")
        try:
            from nemo.collections.asr.models import SortformerEncLabelModel
        except ImportError as exc:
            raise ImportError(
                "Nemotron diarization requires the optional NeMo ASR dependencies; "
                "see docs/cookbook/nemotron_diarization.md"
            ) from exc
        if not hasattr(SortformerEncLabelModel, "_resolve_output_resolution"):
            raise RuntimeError(
                "Installed NeMo lacks Nemotron 3 diarization support. "
                f"Use NVIDIA-NeMo/Speech revision {NEMO_REVISION}; "
                "see docs/cookbook/nemotron_diarization.md"
            )
        checkpoint = resolve_nemo_checkpoint(model_path)
        validate_checkpoint(checkpoint)
        self.model = SortformerEncLabelModel.restore_from(
            restore_path=str(checkpoint), map_location=device, strict=True
        ).eval()
        for name, value in zip(
            (
                "spkcache_len",
                "fifo_len",
                "chunk_len",
                "chunk_right_context",
                "spkcache_update_period",
            ),
            _PROFILES[profile],
        ):
            setattr(self.model.sortformer_modules, name, value)
        self.model._check_streaming_parameters()

    @torch.inference_mode()
    def diarize(self, waveform: np.ndarray) -> DiarizationResult:
        # NeMo mutates preprocessing settings during diarize(); the owning
        # scheduler serializes calls. Each call initializes a fresh speaker cache.
        results = self.model.diarize(
            audio=[waveform],
            sample_rate=SAMPLE_RATE,
            batch_size=1,
            num_workers=0,
            verbose=False,
        )
        if len(results) != 1:
            raise RuntimeError("NeMo returned an unexpected number of recordings")
        return parse_segments(results[0], duration=len(waveform) / SAMPLE_RATE)
