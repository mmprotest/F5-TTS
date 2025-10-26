"""Simple FastAPI server for F5-TTS inference.

This script exposes a HTTP endpoint that accepts only the text to be generated.
The reference audio (``.mp3`` by default) and the corresponding reference text
are provided at server start-up, allowing the server to reuse them for every
request. Responses include the generated audio encoded as base64 alongside
metadata such as the sample rate and random seed that was used.

Example usage::

    python serve_api.py --reference-audio ./ref.mp3 --reference-text ./ref.txt

Once running, send a POST request to ``/infer`` with a JSON body containing the
field ``text`` representing the utterance to synthesize.
"""

from __future__ import annotations

import argparse
import base64
import io
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from f5_tts.api import F5TTS


def _encode_wav(wav, sample_rate: int) -> str:
    """Encode numpy audio data as a base64 wav string."""

    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, wav, sample_rate, format="WAV")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


class InferenceRequest(BaseModel):
    """Payload for text-to-speech inference requests."""

    text: str


def build_app(
    model: str = "F5TTS_v1_Base",
    ckpt_file: str = "",
    vocab_file: str = "",
    ode_method: str = "euler",
    use_ema: bool = True,
    vocoder_local_path: Optional[str] = None,
    device: Optional[str] = None,
    hf_cache_dir: Optional[str] = None,
    reference_audio_path: Path | str = Path("ref.mp3"),
    reference_text: str = "",
) -> FastAPI:
    """Create a FastAPI application configured for inference."""

    app = FastAPI(title="F5-TTS Inference API")

    @lru_cache(maxsize=1)
    def get_model() -> F5TTS:
        return F5TTS(
            model=model,
            ckpt_file=ckpt_file,
            vocab_file=vocab_file,
            ode_method=ode_method,
            use_ema=use_ema,
            vocoder_local_path=vocoder_local_path,
            device=device,
            hf_cache_dir=hf_cache_dir,
        )

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    reference_audio_path = Path(reference_audio_path)

    if not reference_audio_path.exists():
        raise FileNotFoundError(
            f"Reference audio file '{reference_audio_path}' does not exist"
        )

    if not reference_text:
        raise ValueError("Reference text must not be empty")

    @app.post("/infer")
    async def infer(request: InferenceRequest) -> JSONResponse:
        model_instance = get_model()

        wav, sample_rate, _ = model_instance.infer(
            ref_file=str(reference_audio_path),
            ref_text=reference_text,
            gen_text=request.text,
            remove_silence=False,
            seed=None,
        )

        encoded_wav = _encode_wav(wav, sample_rate)

        return JSONResponse(
            {
                "audio_base64": encoded_wav,
                "sample_rate": sample_rate,
                "seed": model_instance.seed,
            }
        )

    return app


def _load_reference_text(path: Path | str) -> str:
    """Read and normalize the reference transcript."""

    text = Path(path).read_text(encoding="utf-8")
    return text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the F5-TTS inference API server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind the server")
    parser.add_argument("--port", type=int, default=8000, help="Port number for the server")
    parser.add_argument("--model", default="F5TTS_v1_Base", help="Model name to load")
    parser.add_argument("--ckpt-file", default="", help="Path to a checkpoint file")
    parser.add_argument("--vocab-file", default="", help="Optional tokenizer vocabulary file")
    parser.add_argument("--ode-method", default="euler", help="ODE method for sampling")
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA weights when loading the model")
    parser.add_argument("--vocoder-local-path", default=None, help="Local path to vocoder weights")
    parser.add_argument("--device", default=None, help="Device to run inference on")
    parser.add_argument("--hf-cache-dir", default=None, help="Hugging Face cache directory")
    parser.add_argument(
        "--reference-audio",
        required=True,
        help="Path to the reference audio file (e.g., an .mp3 recording)",
    )
    parser.add_argument(
        "--reference-text",
        required=True,
        help="Path to a text file containing the reference transcript",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    app = build_app(
        model=args.model,
        ckpt_file=args.ckpt_file,
        vocab_file=args.vocab_file,
        ode_method=args.ode_method,
        use_ema=not args.no_ema,
        vocoder_local_path=args.vocoder_local_path,
        device=args.device,
        hf_cache_dir=args.hf_cache_dir,
        reference_audio_path=Path(args.reference_audio),
        reference_text=_load_reference_text(args.reference_text),
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
