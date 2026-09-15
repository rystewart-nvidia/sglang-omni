# Nemotron 3 Diarization

[NVIDIA Nemotron 3 Diarization preview](https://huggingface.co/nvidia/Nemotron-3-Diarization-preview)
is a Sortformer model that identifies up to eight speakers in a recording,
including overlapping speech. SGLang-Omni serves it through
`/v1/audio/diarizations`, which returns speaker labels and timestamps.
Use a separate ASR model if you also need a transcript.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md).
Use an NVIDIA GPU and Python 3.12, with FFmpeg installed for compressed audio.
NeMo is not required to serve the model.

## Server Configuration

The model runs on one GPU with FP32 weights. By default, it processes one
recording at a time, but you can increase concurrency with `max_concurrency`
(see [Concurrent Requests](#concurrent-requests)).

```bash
sgl-omni serve --config examples/configs/nemotron_diarization.yaml --port 8000
```

The example downloads a pinned checkpoint revision. To use a local checkpoint,
set `model_path` in your YAML configuration:

```yaml
config_cls: NemotronDiarizationPipelineConfig
model_path: /path/to/Nemotron-3-Diarization-preview.nemo
```

You can also point `model_path` to a directory containing that file. Use
`--config` for this model; automatic discovery with `--model-path` alone does
not support the `.nemo` archive.

## Diarize Audio

Upload a recording as multipart form data. The server converts it to mono and
resamples it to 16 kHz.

```bash
curl http://localhost:8000/v1/audio/diarizations \
    -F file=@conversation.wav
```

Example response:

```json
{
  "duration": 3.5,
  "segments": [
    {"start": 0.2, "end": 2.1, "speaker": "speaker_0"},
    {"start": 1.8, "end": 3.4, "speaker": "speaker_1"}
  ]
}
```

Times are seconds from the start of the recording, with 10 ms frame resolution.
Segments are sorted by start time and can overlap when speakers talk at once.
Silence returns an empty `segments` list. Speaker labels belong to each
recording: `speaker_0` in one request is not necessarily the same person as
`speaker_0` in another.

## Request Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `file` | file | required | Audio file uploaded as multipart form data |
| `model` | string | server default | Must match the served model name if provided |
| `response_format` | string | `json` | Only `json` is supported |
| `stream` | boolean | `false` | Only `false` is supported |

`/v1/audio/diarizations` is a SGLang-Omni extension. It rejects transcription
prompts, language selection, speaker-count overrides, and generation parameters.

## Inference Profiles

The default `offline` profile processes audio in chunks while keeping speaker
context across the recording. To use smaller chunks, select `low_latency`:

```bash
sgl-omni serve --config examples/configs/nemotron_diarization.yaml \
    --diarization.factory.profile low_latency --port 8000
```

Both profiles accept a complete recording and return one response after
processing finishes. The `low_latency` setting changes the model's chunking;
it does not enable live audio input or streamed HTTP responses.

## Concurrent Requests

To process multiple recordings at once, increase `max_concurrency`. For example:

```bash
sgl-omni serve --config examples/configs/nemotron_diarization.yaml \
    --diarization.factory.max_concurrency 2 --port 8000
```

Active requests share model weights and use separate speaker caches and CUDA
streams. Additional requests wait in the queue. Increasing concurrency uses more
GPU memory; check memory use and throughput with your recording lengths before
raising the limit. Both inference profiles support this setting.

## Known Limitations

- Up to eight speakers per recording. Audio with more speakers is still accepted,
  but the model cannot assign a separate label to each person.
- Active recordings are limited by `max_concurrency`, which defaults to `1`.
- Disconnecting a client discards its result. An inference call already in
  progress finishes before its worker starts another recording.

## Tests

Run the CPU tests for checkpoint validation, timestamp handling, and the endpoint:

```bash
pytest tests/unit_test/nemotron_diarization tests/unit_test/serve/test_diarizations.py
```

The GPU integration tests in `tests/test_model/test_nemotron_diarization.py`
compare both profiles with NeMo and exercise real HTTP requests. They require a
local checkpoint and the ASR dependencies from
[the pinned NeMo source](https://github.com/NVIDIA-NeMo/Speech/tree/2c1a2f91d64566b5d391b83df42f9ab4cd810adb).

```bash
NEMOTRON_DIARIZATION_CHECKPOINT=/path/to/checkpoint \
  python -m pytest tests/test_model/test_nemotron_diarization.py -q
```

Set `NEMOTRON_DIARIZATION_AUDIO_DIR` to include additional WAV recordings in the tests.
