# SPDX-License-Identifier: Apache-2.0
"""Omni stage boundary for standalone speaker diarization."""

import msgspec
import numpy as np

from sglang_omni.client.types import SamplingParams
from sglang_omni.models.nemotron_diarization.backend import (
    SAMPLE_RATE,
    NemotronDiarizer,
)
from sglang_omni.preprocessing.transcription import resolve_audio_source
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio import AudioDecodeError, load_audio
from sglang_omni.utils.device import resolve_concrete_device


def create_diarization_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    profile: str = "offline",
) -> SimpleScheduler:
    concrete_device = resolve_concrete_device(device, gpu_id)
    diarizer = NemotronDiarizer(model_path, device=concrete_device, profile=profile)
    default_params = SamplingParams().to_dict()
    default_params.pop("max_new_tokens")
    default_params["stream"] = False

    def compute(payload: StagePayload) -> StagePayload:
        if payload.request.metadata.get("task") != "diarization":
            raise ValueError(
                "This model accepts requests through /v1/audio/diarizations"
            )
        unsupported = [
            name
            for name, value in payload.request.params.items()
            if name not in default_params or value != default_params[name]
        ]
        if unsupported:
            raise ValueError(f"Unsupported diarization controls: {sorted(unsupported)}")
        try:
            waveform = load_audio(
                resolve_audio_source(payload),
                source_name="diarization",
                target_sample_rate=SAMPLE_RATE,
            )
        except AudioDecodeError as exc:
            raise ValueError("could not decode the uploaded audio") from exc
        if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
            raise ValueError(
                "could not decode the uploaded audio: empty or non-finite waveform"
            )
        result = diarizer.diarize(waveform)
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={"diarization": msgspec.to_builtins(result)},
        )

    return SimpleScheduler(compute, max_concurrency=1)
