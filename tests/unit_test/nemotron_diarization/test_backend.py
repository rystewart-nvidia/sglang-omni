# SPDX-License-Identifier: Apache-2.0
"""Guard checkpoint-family selection and overlapping, frame-rounded intervals."""

import io
import tarfile

import msgspec
import pytest
import yaml

from sglang_omni.models.nemotron_diarization.backend import (
    parse_segments,
    resolve_nemo_checkpoint,
    validate_checkpoint,
)


def test_segments_preserve_overlap_and_clip_only_the_partial_last_frame():
    result = parse_segments(
        ["1.000 2.510 speaker_7", "0.000 2.000 speaker_0"], duration=2.504
    )
    assert msgspec.to_builtins(result) == {
        "duration": 2.504,
        "segments": [
            {"start": 0.0, "end": 2.0, "speaker": "speaker_0"},
            {"start": 1.0, "end": 2.504, "speaker": "speaker_7"},
        ],
    }


@pytest.mark.parametrize(
    "line",
    [
        "nan 1 speaker_0",
        "0 inf speaker_0",
        "-1 1 speaker_0",
        "1 0 speaker_0",
        "0 1 speaker_8",
        "0 20 speaker_0",
        "0 1",
        "0 0 speaker_0",
    ],
)
def test_invalid_model_segments_are_not_silently_repaired(line):
    with pytest.raises(RuntimeError, match="Invalid NeMo diarization segment"):
        parse_segments([line], duration=2.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"high_resolution": False},
        {"sample_rate": 8000},
        {"target": "other.Model"},
        {"streaming_mode": False},
        {"max_num_of_spks": 4},
        {"preprocessor": {"window_stride": 0.08, "sample_rate": 16000}},
    ],
)
def test_checkpoint_validator_rejects_other_nemo_layouts(tmp_path, overrides):
    config = {
        "target": "nemo.collections.asr.models.sortformer_diar_models.SortformerEncLabelModel",
        "sample_rate": 16000,
        "high_resolution": True,
        "output_subsampling_factor": 1,
        "streaming_mode": True,
        "max_num_of_spks": 8,
        "preprocessor": {"window_stride": 0.01, "sample_rate": 16000},
        "encoder": {
            "self_attention_model": "rope",
            "n_layers": 31,
            "subsampling_factor": 8,
        },
        "sortformer_modules": {"num_spks": 8},
    }
    path = tmp_path / "test.nemo"

    def write_archive(values):
        data = yaml.safe_dump(values).encode()
        with tarfile.open(path, "w") as archive:
            info = tarfile.TarInfo("model_config.yaml")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

    write_archive(config)
    validate_checkpoint(path)
    write_archive({**config, **overrides})
    with pytest.raises(ValueError, match="not the supported"):
        validate_checkpoint(path)


def test_local_checkpoint_directory_does_not_select_an_arbitrary_archive(tmp_path):
    (tmp_path / "unrelated.nemo").touch()
    with pytest.raises(FileNotFoundError, match="Nemotron-3-Diarization-preview.nemo"):
        resolve_nemo_checkpoint(str(tmp_path))


def test_missing_local_checkpoint_does_not_become_a_hub_repository(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing checkpoint"):
        resolve_nemo_checkpoint(str(tmp_path / "missing.nemo"))
