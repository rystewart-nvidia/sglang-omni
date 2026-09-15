# Nemotron 3 Diarization

Serve [NVIDIA Nemotron 3 Diarization preview](https://huggingface.co/nvidia/Nemotron-3-Diarization-preview)
through an offline audio-upload endpoint. The model identifies up to eight
speakers and returns their active intervals, including overlapping speech.
Speaker labels are local to each recording. It does not produce transcripts.

## Install

Use an NVIDIA GPU, Python 3.12, and the normal SGLang-Omni system dependencies
(including FFmpeg for compressed audio). From the Omni checkout, install the
optional NeMo source together with Omni so its exact dependency pins remain
constraints during resolution. Keep the source checkout on persistent storage:

```bash
NEMO_SOURCE=/path/to/persistent/NeMo-Speech
git clone https://github.com/NVIDIA-NeMo/Speech.git "$NEMO_SOURCE"
git -C "$NEMO_SOURCE" checkout 2c1a2f91d64566b5d391b83df42f9ab4cd810adb
uv pip install --prerelease=allow -e . -e "${NEMO_SOURCE}[asr]"
```

This revision includes the RoPE encoder and high-resolution Sortformer output
required by this checkpoint, and declares `lhotse==2.0.0a6`. Released
`nemo-toolkit==3.0.0` cannot load this model. NeMo is imported only when the
diarization stage starts; other models do not require it.

The wrapper and HTTP path were exercised with Torch 2.13.0, Torchaudio 2.11.0,
Torchvision 0.28.0, Transformers 5.12.1, SGLang 0.5.19, and this NeMo source
revision on an H100. NeMo reports version 3.1.0 at this source revision;
this is not a tested release-wheel installation. Prereleases are needed by
the pinned SGLang dependency set and NeMo's Lhotse requirement.

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

The default `offline` profile processes the recording in NeMo's internal
chunks, retaining speaker-cache state across those chunks. A smaller internal
buffer is available with:

```bash
sgl-omni serve --config examples/configs/nemotron_diarization.yaml \
    --diarization.factory.profile low_latency --port 8000
```

Both profiles return a single response after the recording completes. They do
not enable live audio input or streamed HTTP results. Concurrent uploads are
queued; NeMo calls are serialized because its inference helper temporarily
changes model/preprocessor state. Cancelling a request discards its result;
an already-running NeMo inference call finishes before the worker can accept
another recording.

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

`models/nemotron_diarization/backend.py` owns NeMo loading and inference;
`stages.py` owns the Omni request boundary. The client result and HTTP response
contain ordinary speaker intervals with no checkpoint-format details. A later
native implementation can replace the backend while retaining this interface.

## Validation

Run the CPU unit tests with the normal repository test dependencies:

```bash
pytest tests/unit_test/nemotron_diarization tests/unit_test/serve/test_diarizations.py
```

Run the opt-in integration tests with a locally available checkpoint:

```bash
NEMOTRON_DIARIZATION_CHECKPOINT=/path/to/checkpoint \
  python -m pytest tests/test_model/test_nemotron_diarization.py -q
```

These tests launch real HTTP servers for both profiles and compare intervals
with direct NeMo using identical preprocessing. They cover silence, malformed
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
