"""
JANUS voice output layer.

TTS engine abstraction with file-based toggle for cross-process control.
"""

import json
import os
import re
import subprocess
import threading
from abc import ABC, abstractmethod
from pathlib import Path

JANUS_HOME = Path(os.environ.get("JANUS_HOME", os.path.expanduser("~/.janus")))

DEFAULT_CONFIG = {
    "backend": "piper",
    "piper_binary": str(JANUS_HOME / "bin" / "piper"),
    "voice_model": str(JANUS_HOME / "voices" / "en_US-lessac-medium.onnx"),
    "playback": "paplay",
    "pulse_server": "tcp:127.0.0.1:4713",
    "piper_lib_dir": str(JANUS_HOME / "bin"),
}


def voice_enabled(home: Path | None = None) -> bool:
    """Check if voice output is enabled via the toggle file."""
    h = home or JANUS_HOME
    flag = h / "voice_enabled"
    try:
        return flag.read_text().strip() == "1"
    except (FileNotFoundError, PermissionError):
        return False


def set_voice_enabled(enabled: bool, home: Path | None = None) -> None:
    """Write the voice toggle file."""
    h = home or JANUS_HOME
    h.mkdir(parents=True, exist_ok=True)
    (h / "voice_enabled").write_text("1" if enabled else "0")


def load_voice_config(home: Path | None = None) -> dict:
    """Load voice config, falling back to defaults."""
    h = home or JANUS_HOME
    config_path = h / "voice_config.json"
    config = dict(DEFAULT_CONFIG)
    try:
        with open(config_path) as f:
            config.update(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return config


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_ACTION_BLOCK_RE = re.compile(r"<ACTION>[\s\S]*?</ACTION>", re.DOTALL)
_MD_HEADER_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MD_DIVIDER_RE = re.compile(r"^[-*_]{3,}\s*$", re.MULTILINE)
_MD_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_MD_ITALIC_RE = re.compile(r"\*([^*]+)\*")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_MD_NUMBERED_RE = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
# Decorative unicode that LLMs love to scatter around
_DECORATIVE_RE = re.compile(r"[─━═╌┄┈│┃╎╏┆┇╭╮╯╰╔╗╚╝╠╣╦╩╬▀▄█▌▐░▒▓■□▪▫●○◎◉★☆✦✧✴✳✶✷✸✹✺✻✼❖♦♢♠♣♥♡▶►▷◀◄◁△▽⊕⊗⊙⊃⊂∿⊥≈◎→←↑↓⇒⇐⇑⇓]")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return _ANSI_RE.sub("", text)


def clean_for_voice(text: str) -> str:
    """Strip code blocks, markdown, and decorative symbols for TTS.

    The terminal gets the full text. The voice gets clean prose.
    """
    clean = text
    # Remove ACTION blocks first
    clean = _ACTION_BLOCK_RE.sub("", clean)
    # Remove fenced code blocks entirely (these are commands, not prose)
    clean = _CODE_BLOCK_RE.sub("", clean)
    # Replace inline code with just the text inside (often readable)
    clean = _INLINE_CODE_RE.sub(lambda m: m.group(0)[1:-1], clean)
    # Markdown formatting → plain text
    clean = _MD_HEADER_RE.sub("", clean)
    clean = _MD_DIVIDER_RE.sub("", clean)
    clean = _MD_BOLD_RE.sub(r"\1", clean)
    clean = _MD_ITALIC_RE.sub(r"\1", clean)
    clean = _MD_LINK_RE.sub(r"\1", clean)
    clean = _MD_BULLET_RE.sub("", clean)
    clean = _MD_NUMBERED_RE.sub("", clean)
    # Strip decorative unicode
    clean = _DECORATIVE_RE.sub("", clean)
    # Collapse whitespace
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    clean = re.sub(r"  +", " ", clean)
    return clean.strip()


def chunk_sentences(text: str) -> list[str]:
    """Split text into sentence-sized chunks for TTS.

    Strips ANSI codes and LLM formatting, splits on sentence boundaries.
    Returns list of non-empty strings.
    """
    clean = _strip_ansi(text)
    clean = clean_for_voice(clean)
    if not clean:
        return []
    clean = re.sub(r"\n+", " ", clean)
    parts = _SENTENCE_RE.split(clean)
    return [p.strip() for p in parts if p.strip()]


class TTSEngine(ABC):
    """Backend-agnostic TTS interface."""

    @abstractmethod
    def speak(self, text: str) -> None:
        """Speak the given text. May chunk internally for streaming."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Interrupt any ongoing speech."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is ready to use."""
        ...


class PiperTTS(TTSEngine):
    """Piper TTS backend — on-device, aarch64 native.

    Calls the piper binary as a subprocess, pipes output to sox/play.
    Chunks text into sentences and speaks them sequentially in a background thread.
    """

    def __init__(self, config: dict):
        self._piper = config.get("piper_binary", "")
        self._model = config.get("voice_model", "")
        self._playback = config.get("playback", "paplay")
        self._pulse_server = config.get("pulse_server", "tcp:127.0.0.1:4713")
        self._lib_dir = config.get("piper_lib_dir", "")
        self._noise_scale = config.get("noise_scale", 0.9)
        self._noise_w = config.get("noise_w", 1.4)
        self._length_scale = config.get("length_scale", 0.95)
        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None
        self._warned = False
        self._wav_dir = Path(os.environ.get("JANUS_HOME", os.path.expanduser("~/.janus"))) / "tmp"
        self._wav_dir.mkdir(parents=True, exist_ok=True)

    def is_available(self) -> bool:
        return os.path.isfile(self._piper) and os.path.isfile(self._model)

    def speak(self, text: str) -> None:
        if not text or not text.strip():
            return
        if not self.is_available():
            if not self._warned:
                print(f"\033[2m  Voice: Piper not found at {self._piper} — skipping TTS\033[0m")
                self._warned = True
            return
        self.stop()
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._speak_chunks,
            args=(chunk_sentences(text),),
            daemon=True,
        )
        self._thread.start()

    def _speak_chunks(self, chunks: list[str]) -> None:
        env = dict(os.environ)
        if self._lib_dir:
            env["LD_LIBRARY_PATH"] = self._lib_dir
        if self._pulse_server:
            env["PULSE_SERVER"] = self._pulse_server

        for i, chunk in enumerate(chunks):
            if self._stop_flag.is_set():
                break
            try:
                wav_path = str(self._wav_dir / f"chunk_{i}.wav")
                # Piper: synthesize text to wav file
                piper_proc = subprocess.run(
                    [
                        self._piper, "--model", self._model,
                        "--noise_scale", str(self._noise_scale),
                        "--noise_w", str(self._noise_w),
                        "--length_scale", str(self._length_scale),
                        "--output_file", wav_path,
                    ],
                    input=chunk.encode("utf-8"),
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
                if piper_proc.returncode != 0 or self._stop_flag.is_set():
                    break
                # Play the wav file
                subprocess.run(
                    [self._playback, wav_path],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
            except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
                break
            finally:
                # Clean up wav file
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

    def stop(self) -> None:
        self._stop_flag.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def wait(self) -> None:
        """Block until current speech finishes. Useful before next prompt."""
        if self._thread and self._thread.is_alive():
            self._thread.join()
