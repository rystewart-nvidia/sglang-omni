# Nemotron 3 Diarization

Serve [NVIDIA Nemotron 3 Diarization preview](https://huggingface.co/nvidia/Nemotron-3-Diarization-preview)
through an offline audio-upload endpoint. The model identifies up to eight
speakers and returns their active intervals, including overlapping speech.
Speaker labels are local to each recording. It does not produce transcripts.

## Install

Use an NVIDIA GPU, Python 3.12, and the normal SGLang-Omni system dependencies
(including FFmpeg for compressed audio). Install Omni from the checkout:

```bash
uv pip install --prerelease=allow -e .
```

Inference uses native PyTorch modules and does not require NeMo. The loader reads
`model_config.yaml` with `yaml.safe_load` and loads the tensor state dictionary
from `model_weights.ckpt` with `torch.load(weights_only=True)`. It does not
instantiate YAML targets or extract archive paths. Unsupported architecture or
preprocessing settings and missing/unexpected weights fail at startup.

The preview checkpoint is gated. Obtain access on its Hugging Face page and
configure your Hugging Face authentication before starting the server. Its
NVIDIA evaluation license governs model use and disclosure of evaluation results.

## Launch

```bash
sgl-omni serve --config examples/configs/nemotron_diarization.yaml --port 8000
```

The example pins the checkpoint revision and explicitly selects
`NemotronDiarizationPipelineConfig`. Automatic discovery using only
`--model-path` is not supported for this archive layout. In the YAML file,
`model_path` may also name a local `.nemo` file or a directory containing
`Nemotron-3-Diarization-preview.nemo`.

One GPU runs a serial `SimpleScheduler` stage. It restores the checkpoint with
strict weight matching and uses FP32 inference. Incoming audio is decoded,
downmixed, and resampled to 16 kHz using the shared audio utilities.

The default `offline` profile processes the recording in internal
chunks, retaining speaker-cache state across those chunks. A smaller internal
buffer is available with:

```bash
sgl-omni serve --config examples/configs/nemotron_diarization.yaml \
    --diarization.factory.profile low_latency --port 8000
```

Both profiles return a single response after the recording completes. They do
not enable live audio input or streamed HTTP results. Concurrent uploads are
queued to bound GPU memory. Each recording owns fresh speaker-cache state.
Cancelling a request discards its result; an already-running inference call
finishes before the worker can accept another recording.

The model has eight speaker channels. Recordings with nine or more speakers
are outside its supported capacity; the service cannot determine the true
speaker count and does not detect or reject such recordings automatically.

## Request

`POST /v1/audio/diarizations` is an Omni extension. Upload audio using multipart
form data:

```bash
curl http://localhost:8000/v1/audio/diarizations \
    -F file=@conversation.wav
```

The response schema is:

```json
{
  "duration": 3.5,
  "segments": [
    {"start": 0.2, "end": 2.1, "speaker": "speaker_0"},
    {"start": 1.8, "end": 3.4, "speaker": "speaker_1"}
  ]
}
```

This is an illustrative response, not an evaluation result. Times are seconds
from the start of the recording. Intervals are ordered by start time and may
overlap; silence returns an empty `segments` list. The model emits predictions
at 10 ms resolution. Speaker-cache/chunk settings use 80 ms units instead.

Optional form fields are `model` (the exact served model name),
`response_format=json`, and `stream=false`. Transcription prompts, language
selection, generation parameters, and other response formats are unsupported
and rejected. Use a separate ASR model if a transcript is also needed.

## Implementation boundary

`models/nemotron_diarization/backend.py` owns archive loading and interval
postprocessing; `model.py` implements the encoder and speaker head, and
`speaker_cache.py` retains arrival-order speaker context. `stages.py` owns the
Omni request boundary. The client and HTTP schema are unchanged from the NeMo
wrapper baseline.

The inference math is adapted from Apache-2.0 NVIDIA-NeMo/Speech revision
`2c1a2f91d64566b5d391b83df42f9ab4cd810adb`. The implementation supports this
published FP32 checkpoint, with PyTorch FlexAttention and 10 ms output frames.
The cache/FIFO/chunk settings use 80 ms encoder frames:

| Profile | Speaker cache | FIFO | Chunk | Right context | Cache update period |
| --- | --- | --- | --- | --- | --- |
| `offline` | 264 | 40 | 340 | 40 | 300 |
| `low_latency` | 264 | 264 | 9 | 4 | 222 |

## Validation

Run the CPU unit tests with the normal repository test dependencies:

```bash
pytest tests/unit_test/nemotron_diarization tests/unit_test/serve/test_diarizations.py
```

Parity tests use NeMo as a development-only reference. Install the pinned
source in your test environment before running them:

```bash
NEMO_SOURCE=/path/to/persistent/NeMo-Speech
git clone https://github.com/NVIDIA-NeMo/Speech.git "$NEMO_SOURCE"
git -C "$NEMO_SOURCE" checkout 2c1a2f91d64566b5d391b83df42f9ab4cd810adb
uv pip install --prerelease=allow -e . -e "${NEMO_SOURCE}[asr]"
```

Run the opt-in integration tests with a locally available checkpoint:

```bash
NEMOTRON_DIARIZATION_CHECKPOINT=/path/to/checkpoint \
  python -m pytest tests/test_model/test_nemotron_diarization.py -q
```

These tests launch real HTTP servers for both profiles with NeMo imports
blocked in the server processes. They compare native frame probabilities and
HTTP intervals exactly with direct NeMo using identical preprocessing. They cover silence, malformed
and empty audio, stereo/resampling, partial final frames, cache updates in a
67-second recording, and sequential/concurrent request isolation. Set
`NEMOTRON_DIARIZATION_AUDIO_DIR` to include additional permitted WAV recordings.

A passing parity test establishes integration behavior, not diarization
accuracy. For accuracy evaluation, use permitted speaker-annotated recordings,
declare the DER scoring collar and overlap policy, and measure latency and
memory separately. Score recordings beyond eight speakers separately as outside
the supported model capacity. Retain gated model evaluation artifacts outside
public changes; small fixture checks do not establish corpus-wide accuracy or
throughput.
