"""Text-to-speech output via ElevenLabs, falling back to macOS ``say``.

Uses ElevenLabs when both an API key and a voice id are available; on any
failure (or when they're absent) it speaks through the built-in macOS ``say``
command so VIYON always has a voice during development.

The ElevenLabs import is deferred so this module loads without the package.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil

from core import config

logger = logging.getLogger("viyon.speaker")


class Speaker:
    """Speak text aloud, preferring ElevenLabs and falling back to macOS ``say``.

    Args:
        voice_id: ElevenLabs voice id. Defaults to config ``voice.tts_voice_id``.
        api_key: ElevenLabs API key. Defaults to ``$ELEVENLABS_API_KEY``.
        model_id: ElevenLabs TTS model id.
    """

    def __init__(
        self,
        voice_id: str | None = None,
        api_key: str | None = None,
        model_id: str = "eleven_multilingual_v2",
    ) -> None:
        config.load_env()
        self.voice_id = voice_id if voice_id is not None else config.get("voice", "tts_voice_id", "")
        self.api_key = api_key if api_key is not None else os.getenv("ELEVENLABS_API_KEY")
        self.model_id = model_id

    def set_voice(self, voice_id: str) -> None:
        """Change the ElevenLabs voice used for synthesis."""
        self.voice_id = voice_id

    def _choose_backend(self) -> str:
        """Return ``"elevenlabs"`` when a key and voice are set, else ``"say"``."""
        return "elevenlabs" if (self.api_key and self.voice_id) else "say"

    async def say(self, text: str) -> None:
        """Speak ``text``. Tries ElevenLabs, falls back to macOS ``say`` on failure."""
        if not text or not text.strip():
            return
        if self._choose_backend() == "elevenlabs":
            try:
                await asyncio.to_thread(self._speak_elevenlabs, text)
                return
            except Exception as exc:
                logger.warning("ElevenLabs TTS failed (%s); falling back to macOS say.", exc)
        await self._speak_say(text)

    def _speak_elevenlabs(self, text: str) -> None:
        """Synthesize with ElevenLabs and play the audio (runs off the event loop)."""
        from elevenlabs import play
        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(api_key=self.api_key)
        audio = client.text_to_speech.convert(
            voice_id=self.voice_id,
            model_id=self.model_id,
            text=text,
            output_format="mp3_44100_128",
        )
        play(audio)

    async def _speak_say(self, text: str) -> None:
        """Fallback: speak via the macOS ``say`` command."""
        if shutil.which("say") is None:
            logger.warning("macOS `say` not available; cannot speak: %s", text)
            return
        proc = await asyncio.create_subprocess_exec("say", text)
        await proc.wait()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def _demo() -> None:
        speaker = Speaker()
        print(f"Speaking via backend: {speaker._choose_backend()}")
        await speaker.say("VIYON online. All systems nominal.")

    asyncio.run(_demo())
