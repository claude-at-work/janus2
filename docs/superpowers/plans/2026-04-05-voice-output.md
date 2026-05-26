# Voice Output Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add TTS output to JANUS using Piper TTS, toggled via file flag and Termux notification, with `sox`/`play` for audio playback inside proot.

**Architecture:** A `TTSEngine` abstraction in `janus/voice.py` wraps Piper TTS (or future backends). The shell reads a file-based toggle (`~/.janus/voice_enabled`) before speaking. A shell script on the Termux host side flips the toggle via a persistent notification button. Audio plays through `sox`/`play` which is available in the proot environment.

**Tech Stack:** Piper TTS (aarch64 binary), sox/play (audio playback), Termux:API (notification toggle), Python subprocess (Piper invocation)

**Environment facts discovered during planning:**
- `sox`/`play` is available at `/usr/bin/play` inside proot
- `termux-notification` is NOT visible from inside proot (host-side only)
- `numpy` and `torch` are installed
- No test directory existed (created `tests/`)
- Architecture: aarch64

---

## File Structure

| File | New/Modified | Responsibility |
|------|-------------|----------------|
| `janus/voice.py` | New | TTSEngine ABC, PiperTTS backend, sentence chunking, background playback, toggle check |
| `tests/test_voice.py` | New | Unit tests for voice module |
| `janus/shell.py` | Modified | Voice init, speak after LLM/narrate output, `!voice` command |
| `scripts/voice_toggle.sh` | New | Termux host-side notification toggle script |
| `scripts/setup_voice.sh` | New | Downloads Piper binary + default voice model |

Runtime files (not in repo):
- `~/.janus/voice_enabled` — toggle state (`1`/`0`)
- `~/.janus/voice_config.json` — TTS config (backend, paths, playback command)
- `~/.janus/bin/piper` — Piper binary
- `~/.janus/voices/*.onnx` — Voice models

---

### Task 1: Voice toggle and config primitives

The foundation — reading/writing the toggle file and loading config. Everything else depends on this.

**Files:**
- Create: `janus/voice.py`
- Create: `tests/test_voice.py`

- [ ] **Step 1: Write failing tests for toggle and config**

```python
# tests/test_voice.py
import json
import os
import tempfile
import pytest

# We'll monkey-patch the JANUS_HOME so tests don't touch ~/.janus
@pytest.fixture
def voice_home(tmp_path):
    """Provide an isolated directory for voice state files."""
    return tmp_path


def test_voice_disabled_when_file_missing(voice_home):
    from janus.voice import voice_enabled
    # No file exists — should return False
    assert voice_enabled(home=voice_home) is False


def test_voice_enabled_when_file_contains_1(voice_home):
    from janus.voice import voice_enabled
    (voice_home / "voice_enabled").write_text("1")
    assert voice_enabled(home=voice_home) is True


def test_voice_disabled_when_file_contains_0(voice_home):
    from janus.voice import voice_enabled
    (voice_home / "voice_enabled").write_text("0")
    assert voice_enabled(home=voice_home) is False


def test_set_voice_enabled(voice_home):
    from janus.voice import set_voice_enabled, voice_enabled
    set_voice_enabled(True, home=voice_home)
    assert voice_enabled(home=voice_home) is True
    set_voice_enabled(False, home=voice_home)
    assert voice_enabled(home=voice_home) is False


def test_load_config_defaults(voice_home):
    from janus.voice import load_voice_config
    cfg = load_voice_config(home=voice_home)
    assert cfg["backend"] == "piper"
    assert "piper_binary" in cfg
    assert "playback" in cfg


def test_load_config_from_file(voice_home):
    from janus.voice import load_voice_config
    custom = {"backend": "piper", "piper_binary": "/custom/piper", "voice_model": "/custom/voice.onnx", "playback": "play"}
    (voice_home / "voice_config.json").write_text(json.dumps(custom))
    cfg = load_voice_config(home=voice_home)
    assert cfg["piper_binary"] == "/custom/piper"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/janus && python -m pytest tests/test_voice.py -v`
Expected: FAIL — `janus.voice` module does not exist

- [ ] **Step 3: Implement toggle and config functions**

```python
# janus/voice.py
"""
JANUS voice output layer.

TTS engine abstraction with file-based toggle for cross-process control.
"""

import json
import os
from pathlib import Path

JANUS_HOME = Path(os.environ.get("JANUS_HOME", os.path.expanduser("~/.janus")))

DEFAULT_CONFIG = {
    "backend": "piper",
    "piper_binary": str(JANUS_HOME / "bin" / "piper"),
    "voice_model": str(JANUS_HOME / "voices" / "en_US-lessac-medium.onnx"),
    "playback": "play",
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /root/janus && python -m pytest tests/test_voice.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add janus/voice.py tests/test_voice.py
git commit -m "feat(voice): add toggle and config primitives"
```

---

### Task 2: Sentence chunker

Split text into sentence-sized pieces for streaming TTS. This is pure text processing — no audio dependencies.

**Files:**
- Modify: `janus/voice.py`
- Modify: `tests/test_voice.py`

- [ ] **Step 1: Write failing tests for sentence chunking**

Add to `tests/test_voice.py`:

```python
def test_chunk_simple_sentences():
    from janus.voice import chunk_sentences
    text = "Hello there. How are you? I'm fine!"
    chunks = chunk_sentences(text)
    assert chunks == ["Hello there.", "How are you?", "I'm fine!"]


def test_chunk_preserves_long_sentence():
    from janus.voice import chunk_sentences
    text = "This is one long sentence without any terminal punctuation"
    chunks = chunk_sentences(text)
    assert chunks == ["This is one long sentence without any terminal punctuation"]


def test_chunk_handles_abbreviations():
    from janus.voice import chunk_sentences
    text = "Dr. Smith went to Washington. He arrived at 3 p.m. today."
    chunks = chunk_sentences(text)
    # Should not split on Dr. or p.m. — but we accept imperfect splits
    # as long as nothing is lost
    joined = " ".join(chunks)
    assert "Dr." in joined
    assert "Washington" in joined


def test_chunk_empty_string():
    from janus.voice import chunk_sentences
    assert chunk_sentences("") == []


def test_chunk_strips_ansi():
    from janus.voice import chunk_sentences
    text = "\033[92mHello there.\033[0m How are you?"
    chunks = chunk_sentences(text)
    assert chunks == ["Hello there.", "How are you?"]


def test_chunk_multiline():
    from janus.voice import chunk_sentences
    text = "First sentence.\nSecond sentence.\nThird."
    chunks = chunk_sentences(text)
    assert chunks == ["First sentence.", "Second sentence.", "Third."]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/janus && python -m pytest tests/test_voice.py::test_chunk_simple_sentences -v`
Expected: FAIL — `chunk_sentences` not found

- [ ] **Step 3: Implement sentence chunker**

Add to `janus/voice.py`:

```python
import re

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return _ANSI_RE.sub("", text)


def chunk_sentences(text: str) -> list[str]:
    """Split text into sentence-sized chunks for TTS.

    Strips ANSI codes, splits on sentence boundaries (.!? followed by whitespace).
    Returns list of non-empty strings.
    """
    clean = _strip_ansi(text).strip()
    if not clean:
        return []
    # Normalize newlines to spaces for splitting, then split on sentence boundaries
    clean = re.sub(r"\n+", " ", clean)
    parts = _SENTENCE_RE.split(clean)
    return [p.strip() for p in parts if p.strip()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /root/janus && python -m pytest tests/test_voice.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add janus/voice.py tests/test_voice.py
git commit -m "feat(voice): add sentence chunker with ANSI stripping"
```

---

### Task 3: TTSEngine abstraction and PiperTTS backend

The core TTS engine — calls the Piper binary, plays audio via sox/play, runs in a background thread.

**Files:**
- Modify: `janus/voice.py`
- Modify: `tests/test_voice.py`

- [ ] **Step 1: Write failing tests for TTSEngine**

Add to `tests/test_voice.py`:

```python
from unittest.mock import patch, MagicMock
import subprocess


def test_piper_is_available_when_binary_exists(voice_home, tmp_path):
    from janus.voice import PiperTTS
    # Create fake piper binary and model
    piper_bin = tmp_path / "piper"
    piper_bin.write_text("#!/bin/sh\necho fake")
    piper_bin.chmod(0o755)
    model = tmp_path / "voice.onnx"
    model.write_text("fake model")
    config = {
        "piper_binary": str(piper_bin),
        "voice_model": str(model),
        "playback": "play",
    }
    engine = PiperTTS(config)
    assert engine.is_available() is True


def test_piper_not_available_when_binary_missing(voice_home):
    from janus.voice import PiperTTS
    config = {
        "piper_binary": "/nonexistent/piper",
        "voice_model": "/nonexistent/voice.onnx",
        "playback": "play",
    }
    engine = PiperTTS(config)
    assert engine.is_available() is False


def test_piper_speak_calls_subprocess(voice_home, tmp_path):
    from janus.voice import PiperTTS
    piper_bin = tmp_path / "piper"
    piper_bin.write_text("#!/bin/sh\necho fake")
    piper_bin.chmod(0o755)
    model = tmp_path / "voice.onnx"
    model.write_text("fake")
    config = {
        "piper_binary": str(piper_bin),
        "voice_model": str(model),
        "playback": "play",
    }
    engine = PiperTTS(config)
    with patch("janus.voice.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        engine.speak("Hello.")
        # Should have been called — piper piped to play
        assert mock_run.called


def test_piper_speak_skips_empty_text(voice_home, tmp_path):
    from janus.voice import PiperTTS
    config = {
        "piper_binary": "/fake/piper",
        "voice_model": "/fake/voice.onnx",
        "playback": "play",
    }
    engine = PiperTTS(config)
    with patch("janus.voice.subprocess.run") as mock_run:
        engine.speak("")
        assert not mock_run.called


def test_piper_stop_sets_flag(voice_home, tmp_path):
    from janus.voice import PiperTTS
    config = {
        "piper_binary": "/fake/piper",
        "voice_model": "/fake/voice.onnx",
        "playback": "play",
    }
    engine = PiperTTS(config)
    engine.stop()
    assert engine._stop_flag.is_set()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/janus && python -m pytest tests/test_voice.py::test_piper_is_available_when_binary_exists -v`
Expected: FAIL — `PiperTTS` not found

- [ ] **Step 3: Implement PiperTTS**

Add to `janus/voice.py`:

```python
import subprocess
import threading
from abc import ABC, abstractmethod


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
        self._playback = config.get("playback", "play")
        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None
        self._warned = False

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
        # Stop any ongoing speech
        self.stop()
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._speak_chunks,
            args=(chunk_sentences(text),),
            daemon=True,
        )
        self._thread.start()

    def _speak_chunks(self, chunks: list[str]) -> None:
        for chunk in chunks:
            if self._stop_flag.is_set():
                break
            try:
                # Piper reads from stdin, writes wav to stdout
                # Pipe directly into play for streaming playback
                piper_cmd = [
                    self._piper,
                    "--model", self._model,
                    "--output_raw",
                ]
                play_cmd = [
                    self._playback,
                    "-t", "raw",
                    "-r", "22050",
                    "-e", "signed-integer",
                    "-b", "16",
                    "-c", "1",
                    "-",  # read from stdin
                ]
                piper_proc = subprocess.Popen(
                    piper_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                play_proc = subprocess.Popen(
                    play_cmd,
                    stdin=piper_proc.stdout,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                piper_proc.stdin.write(chunk.encode("utf-8"))
                piper_proc.stdin.close()
                piper_proc.stdout.close()
                play_proc.wait()
                piper_proc.wait()
            except (OSError, subprocess.SubprocessError):
                break

    def stop(self) -> None:
        self._stop_flag.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def wait(self) -> None:
        """Block until current speech finishes. Useful before next prompt."""
        if self._thread and self._thread.is_alive():
            self._thread.join()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /root/janus && python -m pytest tests/test_voice.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add janus/voice.py tests/test_voice.py
git commit -m "feat(voice): add TTSEngine ABC and PiperTTS backend"
```

---

### Task 4: Shell integration — `!voice` command and speak hooks

Wire voice into the JANUS REPL: initialization, `!voice` command, and speak hooks after LLM and narrate output.

**Files:**
- Modify: `janus/shell.py`
- Modify: `tests/test_voice.py`

- [ ] **Step 1: Write failing test for shell voice command parsing**

Add to `tests/test_voice.py`:

```python
def test_voice_command_on(voice_home):
    from janus.voice import set_voice_enabled, voice_enabled
    set_voice_enabled(False, home=voice_home)
    set_voice_enabled(True, home=voice_home)
    assert voice_enabled(home=voice_home) is True


def test_voice_command_off(voice_home):
    from janus.voice import set_voice_enabled, voice_enabled
    set_voice_enabled(True, home=voice_home)
    set_voice_enabled(False, home=voice_home)
    assert voice_enabled(home=voice_home) is False
```

(These test the primitives that the shell command will use — the shell integration itself is tested manually since it's an interactive REPL.)

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /root/janus && python -m pytest tests/test_voice.py -v`
Expected: PASS (these use already-implemented functions, just confirming the round-trip)

- [ ] **Step 3: Add voice initialization to shell.py run()**

In `janus/shell.py`, add import at top with the other imports:

```python
from janus.voice import voice_enabled, set_voice_enabled, load_voice_config, PiperTTS
```

In `run()`, after the symlink graph loading block (after line 509), add:

```python
    # ── Voice output ──
    _voice_config = load_voice_config()
    _voice_engine = PiperTTS(_voice_config)
    if _voice_engine.is_available():
        _voice_status = "ready"
    else:
        _voice_status = "unavailable (piper not found)"
    print(f"{C_DIM}  Voice: {_voice_status} | {'on' if voice_enabled() else 'off'}{C_RESET}")
```

- [ ] **Step 4: Add `!voice` command handler**

In `shell.py`, in the special commands section (after the `!notes` block, before `!narrate`), add:

```python
        if user_input.startswith("!voice"):
            voice_arg = user_input[6:].strip()
            if voice_arg == "on":
                set_voice_enabled(True)
                print(f"{C_GREEN}  Voice output enabled.{C_RESET}\n")
            elif voice_arg == "off":
                set_voice_enabled(False)
                _voice_engine.stop()
                print(f"{C_DIM}  Voice output disabled.{C_RESET}\n")
            else:
                status = f"{C_GREEN}on{C_RESET}" if voice_enabled() else f"{C_DIM}off{C_RESET}"
                avail = f"{C_GREEN}ready{C_RESET}" if _voice_engine.is_available() else f"{C_RED}unavailable{C_RESET}"
                print(f"{C_DIM}  Voice: {status} | Engine: {avail}")
                print(f"  Usage: !voice on  |  !voice off{C_RESET}\n")
            continue
```

- [ ] **Step 5: Add speak hooks after LLM response and narrate output**

After LLM response display (after `reply, action = think_display(raw, name=persona_name)` at ~line 692), add:

```python
        # Speak the reply if voice is enabled
        if voice_enabled() and _voice_engine.is_available():
            _voice_engine.speak(reply)
```

After narrate output in narrate mode (after `print(f"{C_MAGENTA}{result}{C_RESET}\n")` at ~line 662), add:

```python
                if voice_enabled() and _voice_engine.is_available():
                    _voice_engine.speak(result)
```

Also after the `!narrate <query>` output (after `print(f"{C_MAGENTA}{result}{C_RESET}\n")` at ~line 630), add:

```python
                if voice_enabled() and _voice_engine.is_available():
                    _voice_engine.speak(result)
```

Before the input prompt at the top of the while loop (before `first = input(active_prefix)` at ~line 521), add a wait so speech finishes before the prompt appears:

```python
        # Wait for any ongoing speech to finish before prompting
        if _voice_engine:
            _voice_engine.wait()
```

- [ ] **Step 6: Add !voice to HELP_TEXT**

In the HELP_TEXT string, add after the `!narrate off` line:

```python
  !voice           - show voice output status
  !voice on        - enable voice output (TTS)
  !voice off       - disable voice output
```

- [ ] **Step 7: Commit**

```bash
git add janus/shell.py tests/test_voice.py
git commit -m "feat(voice): integrate TTS into shell REPL with !voice command"
```

---

### Task 5: Piper setup script

A script to download the Piper binary and a default voice model into `~/.janus/`.

**Files:**
- Create: `scripts/setup_voice.sh`

- [ ] **Step 1: Write the setup script**

```bash
#!/usr/bin/env bash
# setup_voice.sh — Download Piper TTS binary and default voice model
# Run this once to set up voice output for JANUS.

set -euo pipefail

JANUS_HOME="${JANUS_HOME:-$HOME/.janus}"
BIN_DIR="$JANUS_HOME/bin"
VOICE_DIR="$JANUS_HOME/voices"

PIPER_VERSION="2023.11.14-2"
PIPER_URL="https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_aarch64.tar.gz"
VOICE_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
VOICE_NAME="en_US-lessac-medium"

echo "=== JANUS Voice Setup ==="
echo ""

# ── Piper binary ──
if [ -f "$BIN_DIR/piper" ]; then
    echo "  Piper binary already exists at $BIN_DIR/piper"
else
    echo "  Downloading Piper TTS ($PIPER_VERSION)..."
    mkdir -p "$BIN_DIR"
    TMP=$(mktemp -d)
    wget -q --show-progress -O "$TMP/piper.tar.gz" "$PIPER_URL"
    tar xzf "$TMP/piper.tar.gz" -C "$TMP"
    cp "$TMP/piper/piper" "$BIN_DIR/piper"
    # Copy piper libs if present
    if [ -d "$TMP/piper/lib" ]; then
        cp -r "$TMP/piper/lib" "$BIN_DIR/"
    fi
    # Copy espeak-ng-data if present (piper needs it)
    if [ -d "$TMP/piper/espeak-ng-data" ]; then
        cp -r "$TMP/piper/espeak-ng-data" "$BIN_DIR/"
    fi
    chmod +x "$BIN_DIR/piper"
    rm -rf "$TMP"
    echo "  Installed to $BIN_DIR/piper"
fi

# ── Voice model ──
if [ -f "$VOICE_DIR/${VOICE_NAME}.onnx" ]; then
    echo "  Voice model already exists at $VOICE_DIR/${VOICE_NAME}.onnx"
else
    echo "  Downloading voice model (${VOICE_NAME})..."
    mkdir -p "$VOICE_DIR"
    wget -q --show-progress -O "$VOICE_DIR/${VOICE_NAME}.onnx" \
        "${VOICE_BASE}/${VOICE_NAME}.onnx"
    wget -q --show-progress -O "$VOICE_DIR/${VOICE_NAME}.onnx.json" \
        "${VOICE_BASE}/${VOICE_NAME}.onnx.json"
    echo "  Installed to $VOICE_DIR/"
fi

# ── Write default config if none exists ──
CONFIG="$JANUS_HOME/voice_config.json"
if [ ! -f "$CONFIG" ]; then
    cat > "$CONFIG" <<CONF
{
  "backend": "piper",
  "piper_binary": "$BIN_DIR/piper",
  "voice_model": "$VOICE_DIR/${VOICE_NAME}.onnx",
  "playback": "play"
}
CONF
    echo "  Config written to $CONFIG"
fi

echo ""
echo "  Done. Test with:"
echo "    echo 'Hello from JANUS' | $BIN_DIR/piper --model $VOICE_DIR/${VOICE_NAME}.onnx --output_raw | play -t raw -r 22050 -e signed-integer -b 16 -c 1 -"
echo ""
```

- [ ] **Step 2: Make executable and test syntax**

Run: `chmod +x /root/janus/scripts/setup_voice.sh && bash -n /root/janus/scripts/setup_voice.sh`
Expected: No syntax errors

- [ ] **Step 3: Commit**

```bash
git add scripts/setup_voice.sh
git commit -m "feat(voice): add Piper TTS setup script"
```

---

### Task 6: Termux notification toggle

A script that runs on the Termux host side to toggle voice via a persistent notification button.

**Files:**
- Create: `scripts/voice_toggle.sh`

- [ ] **Step 1: Write the toggle script**

```bash
#!/data/data/com.termux/files/usr/bin/bash
# voice_toggle.sh — Toggle JANUS voice from a Termux notification button.
# This script runs on the Termux HOST side (not inside proot).
# It reads/writes ~/.janus/voice_enabled which is visible from both sides.

VOICE_FILE="$HOME/.janus/voice_enabled"
TERMUX_NOTIF="termux-notification"

# Read current state
if [ -f "$VOICE_FILE" ] && [ "$(cat "$VOICE_FILE")" = "1" ]; then
    # Currently on → turn off
    echo "0" > "$VOICE_FILE"
    STATE="OFF"
else
    # Currently off → turn on
    mkdir -p "$(dirname "$VOICE_FILE")"
    echo "1" > "$VOICE_FILE"
    STATE="ON"
fi

# Update notification
$TERMUX_NOTIF \
    --id janus-voice \
    --title "JANUS Voice: $STATE" \
    --content "Tap to toggle" \
    --ongoing \
    --button1 "Toggle" \
    --button1-action "bash $0"
```

- [ ] **Step 2: Write the notification setup helper**

Add to `scripts/voice_toggle.sh` at the bottom, guarded by an `--init` flag:

```bash
# ── Setup: create the initial notification ──
# Run once: bash voice_toggle.sh --init
if [ "${1:-}" = "--init" ]; then
    mkdir -p "$(dirname "$VOICE_FILE")"
    echo "0" > "$VOICE_FILE"
    $TERMUX_NOTIF \
        --id janus-voice \
        --title "JANUS Voice: OFF" \
        --content "Tap to toggle" \
        --ongoing \
        --button1 "Toggle" \
        --button1-action "bash $0"
    echo "JANUS voice notification created."
fi
```

- [ ] **Step 3: Commit**

```bash
git add scripts/voice_toggle.sh
git commit -m "feat(voice): add Termux notification toggle script"
```

---

### Task 7: Integration smoke test

Verify the full pipeline works end-to-end (requires Piper to be installed via `setup_voice.sh`).

**Files:**
- Modify: `tests/test_voice.py`

- [ ] **Step 1: Add integration test (skips if Piper not installed)**

Add to `tests/test_voice.py`:

```python
@pytest.mark.skipif(
    not os.path.isfile(os.path.expanduser("~/.janus/bin/piper")),
    reason="Piper not installed — run scripts/setup_voice.sh first"
)
def test_piper_speaks_real_audio():
    """Integration test: actually run Piper and play audio."""
    from janus.voice import load_voice_config, PiperTTS
    config = load_voice_config()
    engine = PiperTTS(config)
    assert engine.is_available()
    # This will actually produce audio — short text to keep it brief
    engine.speak("JANUS voice test.")
    engine.wait()
    # If we get here without exception, it worked
```

- [ ] **Step 2: Run the full test suite**

Run: `cd /root/janus && python -m pytest tests/test_voice.py -v`
Expected: All unit tests PASS, integration test SKIPPED (unless Piper is installed)

- [ ] **Step 3: Commit**

```bash
git add tests/test_voice.py
git commit -m "test(voice): add integration smoke test for Piper TTS"
```

---

### Task 8: Run setup and end-to-end verification

Actually install Piper and test the whole thing.

- [ ] **Step 1: Run the setup script**

Run: `bash /root/janus/scripts/setup_voice.sh`
Expected: Piper binary and voice model downloaded to `~/.janus/`

- [ ] **Step 2: Test Piper directly from the command line**

Run: `echo "Hello from JANUS" | ~/.janus/bin/piper --model ~/.janus/voices/en_US-lessac-medium.onnx --output_raw | play -t raw -r 22050 -e signed-integer -b 16 -c 1 -`
Expected: Audio plays through device speaker. If it fails, check error output — likely candidates are dynamic linker issues (piper in proot) or audio routing (play in proot).

- [ ] **Step 3: Run integration test**

Run: `cd /root/janus && python -m pytest tests/test_voice.py -v`
Expected: All tests PASS including the integration test

- [ ] **Step 4: Test via JANUS shell**

Run: `cd /root/janus && python -m janus`
Then: type `!voice on`, say something, verify audio plays.
Then: type `!voice off`, say something, verify no audio.
Then: type `!voice`, verify status display.

- [ ] **Step 5: Document any proot-specific issues**

If Piper has dynamic linker issues in proot, document the workaround (e.g., `LD_LIBRARY_PATH` or using `piper-tts` Python package instead). Update `voice_config.json` or `setup_voice.sh` as needed.

---

## Dependency Graph

```
Task 1 (toggle + config)
    ↓
Task 2 (chunker)
    ↓
Task 3 (PiperTTS engine)
    ↓
Task 4 (shell integration)
    ↓
Task 5 (setup script)  ←── independent, but needed before Task 8
    ↓
Task 6 (notification toggle)  ←── independent
    ↓
Task 7 (integration test)
    ↓
Task 8 (end-to-end verification)
```

Tasks 5 and 6 are independent of each other and could be done in parallel. Everything else is sequential.
