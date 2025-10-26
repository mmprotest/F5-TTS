"""Gradio interface for running F5-TTS inference locally.

This utility mirrors the configuration options from ``serve_api.py`` but exposes
an interactive web UI instead of a REST API. Users can upload a reference audio
clip, provide the matching transcript, and enter new text to synthesize. The
resulting speech is streamed back to the browser for immediate playback.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from typing import Optional, Tuple

import gradio as gr
import numpy as np

from f5_tts.api import F5TTS


@lru_cache(maxsize=1)
def _get_model(
    model: str,
    ckpt_file: str,
    vocab_file: str,
    ode_method: str,
    use_ema: bool,
    vocoder_local_path: Optional[str],
    device: Optional[str],
    hf_cache_dir: Optional[str],
) -> F5TTS:
    """Load and cache a single ``F5TTS`` instance for the UI session."""

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


def synthesize(
    reference_audio: Optional[str],
    reference_text: str,
    target_text: str,
    seed: Optional[float],
    remove_silence: bool,
    model: str,
    ckpt_file: str,
    vocab_file: str,
    ode_method: str,
    use_ema: bool,
    vocoder_local_path: Optional[str],
    device: Optional[str],
    hf_cache_dir: Optional[str],
) -> Tuple[int, np.ndarray]:
    """Run inference and return audio suitable for ``gr.Audio`` playback."""

    if not reference_audio:
        raise gr.Error("Please upload a reference audio sample (wav format recommended).")
    if not reference_text.strip():
        raise gr.Error("Reference text cannot be empty.")
    if not target_text.strip():
        raise gr.Error("Please enter text to synthesize.")

    seed_value: Optional[int]
    if seed is None or seed == "" or (isinstance(seed, float) and np.isnan(seed)):
        seed_value = None
    else:
        try:
            seed_value = int(seed)
        except (TypeError, ValueError) as exc:
            raise gr.Error("Seed must be an integer value.") from exc

    model_instance = _get_model(
        model=model,
        ckpt_file=ckpt_file,
        vocab_file=vocab_file,
        ode_method=ode_method,
        use_ema=use_ema,
        vocoder_local_path=vocoder_local_path,
        device=device,
        hf_cache_dir=hf_cache_dir,
    )

    wav, sample_rate, _ = model_instance.infer(
        ref_file=reference_audio,
        ref_text=reference_text,
        gen_text=target_text,
        remove_silence=remove_silence,
        seed=seed_value,
    )

    audio = np.asarray(wav, dtype=np.float32)
    return sample_rate, audio


def build_interface(args: argparse.Namespace) -> gr.Blocks:
    """Create the Gradio Blocks UI for text-to-speech synthesis."""

    with gr.Blocks(title="F5-TTS Gradio Demo") as demo:
        gr.Markdown(
            """
            # F5-TTS Interactive Demo

            Upload a reference audio file and its transcript, then provide new text
            to synthesize using the same voice.
            """
        )

        with gr.Row():
            reference_audio = gr.Audio(
                label="Reference Audio",
                sources=["upload"],
                type="filepath",
                interactive=True,
            )
            seed_input = gr.Number(label="Seed", value=None)
        reference_text = gr.Textbox(label="Reference Text", lines=3)
        target_text = gr.Textbox(label="Target Text", lines=3)
        remove_silence = gr.Checkbox(label="Remove Silence", value=False)

        output_audio = gr.Audio(label="Generated Audio", type="numpy")
        submit = gr.Button("Synthesize")

        submit.click(
            synthesize,
            inputs=[
                reference_audio,
                reference_text,
                target_text,
                seed_input,
                remove_silence,
                gr.State(args.model),
                gr.State(args.ckpt_file),
                gr.State(args.vocab_file),
                gr.State(args.ode_method),
                gr.State(not args.no_ema),
                gr.State(args.vocoder_local_path),
                gr.State(args.device),
                gr.State(args.hf_cache_dir),
            ],
            outputs=[output_audio],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Gradio UI for F5-TTS inference")
    parser.add_argument("--model", default="F5TTS_v1_Base", help="Model name to load")
    parser.add_argument("--ckpt-file", default="", help="Path to a checkpoint file")
    parser.add_argument("--vocab-file", default="", help="Optional tokenizer vocabulary file")
    parser.add_argument("--ode-method", default="euler", help="ODE method for sampling")
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA weights when loading the model")
    parser.add_argument("--vocoder-local-path", default=None, help="Local path to vocoder weights")
    parser.add_argument("--device", default=None, help="Device to run inference on")
    parser.add_argument("--hf-cache-dir", default=None, help="Hugging Face cache directory")
    parser.add_argument("--share", action="store_true", help="Share the Gradio app publicly")
    parser.add_argument("--server-name", default="127.0.0.1", help="Host address for the Gradio server")
    parser.add_argument("--server-port", type=int, default=7860, help="Port number for the Gradio server")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_interface(args)
    demo.launch(share=args.share, server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
