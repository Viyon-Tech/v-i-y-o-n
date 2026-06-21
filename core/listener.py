"""Microphone capture + energy VAD + Whisper transcription.

Records from the mic until ~1.2s of trailing silence, then transcribes with
mlx-whisper (Apple Silicon native), falling back to faster-whisper if
mlx-whisper isn't importable.

All heavy imports (numpy, sounddevice, mlx_whisper, faster_whisper) are
deferred so this module imports cleanly without them. ``transcribe_file`` only
needs a Whisper backend — no mic or numpy — which keeps it easy to test.
"""

from __future__ import annotations

import logging
import asyncio
from pathlib import Path

from core import config

logger = logging.getLogger("viyon.listener")


class Listener:
    """Capture a single spoken utterance from the mic and transcribe it.

    Args:
        model: Whisper model name/repo. Defaults to config ``voice.whisper_model``
            (or ``"small"``). A bare size like ``"small"`` is expanded to an
            ``mlx-community/whisper-<size>`` repo for mlx-whisper.
        sample_rate: Capture sample rate (Whisper expects 16 kHz).
        block_ms: VAD analysis block size in milliseconds.
        silence_duration: Trailing silence (seconds) that ends an utterance.
        silence_threshold: RMS energy below which a block counts as silence.
        max_seconds: Hard cap on a single recording.
        start_timeout: Give up if no speech starts within this many seconds.
    """

    def __init__(
        self,
        model: str | None = None,
        sample_rate: int = 16000,
        block_ms: int = 30,
        silence_duration: float = 1.2,
        silence_threshold: float = 0.015,
        max_seconds: float = 15.0,
        start_timeout: float = 8.0,
    ) -> None:
        self.model = model or config.get("voice", "whisper_model", "small")
        self.sample_rate = sample_rate
        self.block = int(sample_rate * block_ms / 1000)
        self.silence_duration = silence_duration
        self.silence_threshold = silence_threshold
        self.max_seconds = max_seconds
        self.start_timeout = start_timeout

    async def listen(self) -> str:
        """Record one utterance and return its transcript (off the event loop)."""
        audio = await asyncio.to_thread(self._record)
        if audio is None or len(audio) == 0:
            return ""
        return await asyncio.to_thread(self._transcribe, audio)

    def transcribe_file(self, path: str | Path) -> str:
        """Transcribe an existing audio file. Handy for tests and replay."""
        return self._transcribe(str(path))

    # -- recording ----------------------------------------------------------

    def _record(self):
        """Capture mic audio with energy-based VAD; return a float32 mono array."""
        import numpy as np
        import sounddevice as sd

        block_sec = self.block / self.sample_rate
        needed_silence = max(1, int(self.silence_duration / block_sec))
        max_blocks = int(self.max_seconds / block_sec)
        start_blocks = int(self.start_timeout / block_sec)

        frames: list = []
        silent_blocks = 0
        speech_started = False

        with sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32", blocksize=self.block
        ) as stream:
            for i in range(max_blocks):
                data, _ = stream.read(self.block)
                chunk = np.asarray(data, dtype=np.float32).reshape(-1)
                rms = float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0

                if rms >= self.silence_threshold:
                    speech_started = True
                    silent_blocks = 0
                    frames.append(chunk)
                elif speech_started:
                    silent_blocks += 1
                    frames.append(chunk)
                    if silent_blocks >= needed_silence:
                        break
                elif i >= start_blocks:
                    break  # no speech ever started

        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames)

    # -- transcription ------------------------------------------------------

    def _mlx_repo(self) -> str:
        """Expand a bare model size into an mlx-community HF repo."""
        return self.model if "/" in self.model else f"mlx-community/whisper-{self.model}"

    def _transcribe(self, audio) -> str:
        """Transcribe a numpy array or file path; mlx-whisper first, else faster-whisper."""
        try:
            import mlx_whisper

            result = mlx_whisper.transcribe(audio, path_or_hf_repo=self._mlx_repo())
            return (result.get("text") or "").strip()
        except ImportError:
            logger.info("mlx-whisper unavailable; trying faster-whisper.")
        except Exception as exc:
            logger.warning("mlx-whisper failed (%s); trying faster-whisper.", exc)

        from faster_whisper import WhisperModel

        fw = WhisperModel(self.model)
        segments, _ = fw.transcribe(audio)
        return "".join(segment.text for segment in segments).strip()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def _demo() -> None:
        listener = Listener()
        print(f"Recording (model={listener.model})... speak now.")
        text = await listener.listen()
        print(f"📝 Heard: {text!r}")

    asyncio.run(_demo())
