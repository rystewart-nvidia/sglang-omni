# SPDX-License-Identifier: Apache-2.0
"""Offline diarization using NeMo's internally chunked Sortformer inference."""

from typing import ClassVar

from sglang_omni.config import FactoryArgs, PipelineConfig, StageConfig


class NemotronDiarizationPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "SortformerEncLabelModel"
    requires_model_capabilities: ClassVar[bool] = True

    stages: list[StageConfig] = [
        StageConfig(
            name="diarization",
            process="diarization",
            factory_path="sglang_omni.models.nemotron_diarization.stages.create_diarization_executor",
            factory=FactoryArgs(profile="offline"),
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = NemotronDiarizationPipelineConfig
