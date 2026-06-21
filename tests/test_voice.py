"""Tests for the voice pipeline (wake word, listener, speaker).

Audio is hard to unit-test, so these tests focus on the parts that don't need a
mic or a network: backend selection, graceful fallbacks, and ``transcribe_file``
when a Whisper backend happens to be installed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from core.listener import Listener
from core.speaker import Speaker
from core.wake_word import WakeWord

FIXTURE_WAV = Path(__file__).parent / "fixtures" / "tiny.wav"


# -- Speaker -----------------------------------------------------------------

def test_say_fallback_selected_without_key():
    """With no API key, the macOS `say` fallback is chosen."""
    assert Speaker(api_key="", voice_id="some-voice")._choose_backend() == "say"


def test_say_fallback_selected_without_voice():
    """A key but no voice id still falls back to `say`."""
    assert Speaker(api_key="k", voice_id="")._choose_backend() == "say"


def test_elevenlabs_selected_with_key_and_voice():
    """With both a key and a voice id, ElevenLabs is chosen."""
    assert Speaker(api_key="k", voice_id="v")._choose_backend() == "elevenlabs"


async def test_say_invokes_say_binary(monkeypatch):
    """The fallback path shells out to the macOS `say` command with the text."""
    recorded: dict = {}

    class FakeProc:
        async def wait(self):
            return 0

    async def fake_exec(*args, **kwargs):
        recorded["args"] = args
        return FakeProc()

    monkeypatch.setattr("core.speaker.shutil.which", lambda _: "/usr/bin/say")
    monkeypatch.setattr("core.speaker.asyncio.create_subprocess_exec", fake_exec)

    await Speaker(api_key="").say("hello viyon")

    assert recorded["args"] == ("say", "hello viyon")


async def test_say_ignores_empty_text(monkeypatch):
    """Empty/whitespace text never reaches a backend."""
    called = False

    async def fake_exec(*args, **kwargs):  # pragma: no cover - must not run
        nonlocal called
        called = True

    monkeypatch.setattr("core.speaker.asyncio.create_subprocess_exec", fake_exec)
    await Speaker(api_key="").say("   ")
    assert called is False


def test_set_voice_updates_backend_choice():
    """set_voice flips a key-only speaker over to ElevenLabs."""
    speaker = Speaker(api_key="k", voice_id="")
    assert speaker._choose_backend() == "say"
    speaker.set_voice("v")
    assert speaker._choose_backend() == "elevenlabs"


# -- WakeWord ----------------------------------------------------------------

def test_wake_word_fallback_without_key():
    """Missing PICOVOICE_ACCESS_KEY → no engine, clear fallback reason."""
    ww = WakeWord(access_key="")
    assert ww._init_engine() is None
    assert "PICOVOICE_ACCESS_KEY" in ww.fallback_reason


def test_wake_word_fallback_without_keyword_files():
    """A key but no .ppn files → fall back and explain why."""
    ww = WakeWord(access_key="fake-key", keyword_paths=[])
    assert ww._init_engine() is None
    assert ".ppn" in ww.fallback_reason


async def test_wake_word_keyboard_fallback_resolves(monkeypatch):
    """In fallback mode, wait_for_wake resolves once 'Enter' is pressed."""
    monkeypatch.setattr("core.wake_word.input", lambda *a, **k: "", raising=False)
    fired = []
    ww = WakeWord(access_key="", on_wake=lambda: fired.append(True))
    await asyncio.wait_for(ww.wait_for_wake(), timeout=5)
    assert fired == [True]


# -- Listener ----------------------------------------------------------------

def test_listener_default_model_from_config():
    """Listener picks up the configured Whisper model."""
    assert isinstance(Listener().model, str) and Listener().model


def test_mlx_repo_expansion():
    """Bare sizes expand to mlx-community repos; full repos pass through."""
    assert Listener(model="small")._mlx_repo() == "mlx-community/whisper-small"
    full = "mlx-community/whisper-large-v3-turbo"
    assert Listener(model=full)._mlx_repo() == full


def test_transcribe_file():
    """transcribe_file returns a string for the bundled wav (needs a backend)."""
    have_backend = False
    for mod in ("mlx_whisper", "faster_whisper"):
        try:
            __import__(mod)
            have_backend = True
            break
        except ImportError:
            continue
    if not have_backend:
        pytest.skip("no Whisper backend installed (mlx-whisper / faster-whisper)")

    assert FIXTURE_WAV.exists(), "bundled test wav is missing"
    text = Listener(model="tiny").transcribe_file(FIXTURE_WAV)
    assert isinstance(text, str)
