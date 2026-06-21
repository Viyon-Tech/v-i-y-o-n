"""Wake-word detection for "VIYON" via pvporcupine (Picovoice).

Listens continuously for the VIYON wake words. If Picovoice can't be used
(no access key, no trained ``.ppn`` keyword files, or the engine fails to
start), it degrades to a "press Enter to talk" mode and logs a clear warning,
so development is never blocked on Picovoice setup.

Heavy audio imports (pvporcupine, sounddevice) are deferred into methods so
this module imports cleanly even when those packages aren't installed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import struct
from pathlib import Path

from core import config

logger = logging.getLogger("viyon.wake_word")

# The phrases VIYON wakes on. These are custom wake words, so using Picovoice
# requires a trained ``.ppn`` file per phrase (Picovoice console). Without them
# we fall back to keyboard mode.
DEFAULT_KEYWORDS = ("VIYON", "Wake Up, Daddy's home")
KEYWORD_DIR = Path("~/.viyon/keywords").expanduser()


class WakeWord:
    """Continuously listen for a VIYON wake word.

    Args:
        access_key: Picovoice access key. Defaults to ``$PICOVOICE_ACCESS_KEY``.
        keyword_paths: Paths to ``.ppn`` keyword files. Defaults to scanning
            ``~/.viyon/keywords/``.
        sensitivities: Per-keyword detection sensitivity in ``[0, 1]``.
        on_wake: Optional callback invoked when a wake word is detected.
    """

    def __init__(
        self,
        access_key: str | None = None,
        keyword_paths: list[str] | None = None,
        sensitivities: list[float] | None = None,
        on_wake=None,
    ) -> None:
        config.load_env()
        self.access_key = access_key if access_key is not None else os.getenv("PICOVOICE_ACCESS_KEY")
        self.keyword_paths = list(keyword_paths) if keyword_paths else self._discover_keywords()
        self.sensitivities = sensitivities
        self.on_wake = on_wake
        self.fallback_reason: str | None = None
        self._stop = asyncio.Event()

    @staticmethod
    def _discover_keywords() -> list[str]:
        """Return any ``.ppn`` keyword files bundled under ~/.viyon/keywords/."""
        if KEYWORD_DIR.is_dir():
            return [str(p) for p in sorted(KEYWORD_DIR.glob("*.ppn"))]
        return []

    def _init_engine(self):
        """Create a Porcupine instance, or record a fallback reason and return None."""
        if not self.access_key:
            self.fallback_reason = "PICOVOICE_ACCESS_KEY not set"
            return None
        if not self.keyword_paths:
            self.fallback_reason = (
                f"no wake-word (.ppn) files in {KEYWORD_DIR} — "
                f"train {DEFAULT_KEYWORDS[0]!r} in the Picovoice console"
            )
            return None
        try:
            import pvporcupine

            sensitivities = self.sensitivities or [0.5] * len(self.keyword_paths)
            return pvporcupine.create(
                access_key=self.access_key,
                keyword_paths=self.keyword_paths,
                sensitivities=sensitivities,
            )
        except Exception as exc:  # ImportError or Porcupine runtime errors
            self.fallback_reason = f"porcupine init failed: {exc}"
            return None

    async def wait_for_wake(self) -> None:
        """Resolve once a wake word is detected (or Enter is pressed in fallback)."""
        self._stop.clear()
        engine = self._init_engine()
        if engine is None:
            logger.warning(
                "Wake word disabled (%s); using keyboard fallback.", self.fallback_reason
            )
            await self._fallback_wait()
        else:
            try:
                await asyncio.to_thread(self._blocking_detect, engine)
            finally:
                engine.delete()
        if self.on_wake is not None:
            self.on_wake()

    async def _fallback_wait(self) -> None:
        """Keyboard fallback: block until the user presses Enter."""
        await asyncio.to_thread(input, "🎙️  [VIYON] Press Enter to talk... ")

    def _blocking_detect(self, engine) -> None:
        """Read mic frames and return when the wake word fires (runs off-loop)."""
        import sounddevice as sd

        with sd.RawInputStream(
            samplerate=engine.sample_rate,
            blocksize=engine.frame_length,
            dtype="int16",
            channels=1,
        ) as stream:
            while not self._stop.is_set():
                data, _ = stream.read(engine.frame_length)
                pcm = struct.unpack_from("h" * engine.frame_length, data)
                if engine.process(pcm) >= 0:
                    return

    def stop(self) -> None:
        """Request the detection loop to stop at the next frame."""
        self._stop.set()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def _demo() -> None:
        ww = WakeWord()
        print("Listening for wake word... (say 'VIYON' or press Enter in fallback mode)")
        await ww.wait_for_wake()
        print("✅ Wake word detected — VIYON is listening.")

    asyncio.run(_demo())
