# JANUS Voice Output Layer — Design Spec

**Date:** 2026-04-05
**Phase:** 1 of 2 (TTS output only; STT input with wake-word is phase 2)
**Status:** Draft

## Problem

JANUS output is text-only. Getting responses read aloud currently requires manually copying text into Speechify — a $140/year app with no API access. This breaks flow, especially in hands-dirty / hands-free scenarios.

## Goal

Add a TTS output layer to JANUS that:
- Speaks responses aloud using on-device inference (Piper TTS)
- Toggles via a single tap on a Termux notification button
- Keeps text output unchanged — voice is additive, not a replacement
- Uses a backend-agnostic interface so the TTS engine can be swapped later (local or cloud)

## Constraints

- **Device:** Pixel 7 Pro, Tensor G2, aarch64, high RAM
- **Environment:** Termux proot (Kali daily driver)
- **Termux:API:** Installed via F-Droid on host Termux; accessible from proot via host binary path
- **Sovereignty direction:** Build local-first, but don't lock out external backends
- **Resource budget:** Piper is lightweight (~50-80MB model, fast ARM inference via onnxruntime); should not compete meaningfully with Ollama for resources

## Architecture

### Components

#### 1. `janus/voice.py` — TTS engine wrapper

The public interface:

```python
class TTSEngine:
    """Backend-agnostic TTS interface."""
    def speak(self, text: str) -> None: ...
    def stop(self) -> None: ...
    def is_available(self) -> bool: ...

class PiperTTS(TTSEngine):
    """Piper TTS backend — on-device, aarch64 native."""
    ...
```

Responsibilities:
- Accept text, produce and play audio
- Chunk long text into sentence-sized pieces for streaming playback (start speaking before full response is generated)
- Run audio playback in a background thread so the shell isn't blocked
- Provide a `stop()` method to interrupt speech (for the toggle or new input)
- Handle missing binary / model gracefully (warn once, degrade to text-only)

Chunking strategy: split on sentence boundaries (`.` `!` `?` followed by whitespace). Feed chunks to Piper sequentially, playing each as soon as it's rendered. This gives a streaming feel without requiring actual streaming TTS.

#### 2. Voice toggle state

A file-based flag at `~/.janus/voice_enabled`:
- File exists and contains `1` → voice on
- File missing or contains `0` → voice off

Why a file and not in-memory state: the Termux notification action runs as a separate process. It can't reach into the JANUS process to flip a boolean. A file is the simplest IPC that works across the proot boundary.

The shell checks this file before each `speak()` call. Cost is negligible (single stat + read of a 1-byte file).

Default: off. User taps to enable.

#### 3. Termux notification toggle — `janus/voice_toggle.sh`

A shell script that:
1. Reads current state from `~/.janus/voice_enabled`
2. Flips it
3. Writes new state
4. Updates the notification to reflect current state ("Voice: ON" / "Voice: OFF")

Called by a persistent Termux notification action button. Setup:

```bash
# Create persistent notification with toggle action
termux-notification \
  --id janus-voice \
  --title "JANUS Voice" \
  --content "Voice: OFF" \
  --button1 "Toggle" \
  --button1-action "bash /path/to/voice_toggle.sh"
```

The Termux:API binary path from inside proot will need to be resolved. Likely `/data/data/com.termux/files/usr/bin/termux-notification` or accessed via a symlink/wrapper.

#### 4. Shell integration points

Three insertions in `shell.py`:

**a. Initialization (in `run()`, after banner):**
- Import voice module
- Initialize TTS engine
- Set up Termux notification (if available)
- Print voice status in startup info

**b. After LLM response display (~line 692):**
```python
reply, action = think_display(raw, name=persona_name)
# NEW: speak if voice enabled
if voice_enabled():
    voice.speak(reply)
```

**c. After narrate output (~line 662):**
```python
if result:
    print(f"{C_MAGENTA}{result}{C_RESET}\n")
    # NEW: speak narration if voice enabled
    if voice_enabled():
        voice.speak(result)
```

**d. Add `!voice` command as fallback:**
For when the notification isn't available or the user prefers typing:
```
!voice        — show status
!voice on     — enable
!voice off    — disable
```

### Audio playback

Options in order of preference:
1. `termux-media-player` (via Termux:API — plays wav/mp3 natively on Android)
2. `play` from sox (if installed in proot)
3. `aplay` from alsa-utils (if available)
4. Write to file and let user handle (last resort)

Piper outputs wav by default. `termux-media-player` is the natural fit since Termux:API is already installed.

### Piper TTS installation

Piper distributes prebuilt aarch64 binaries. Installation:

```bash
# Download piper binary
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz
tar xzf piper_arm64.tar.gz -C ~/.janus/bin/

# Download voice model
mkdir -p ~/.janus/voices
wget -O ~/.janus/voices/en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget -O ~/.janus/voices/en_US-lessac-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

Default voice: `en_US-lessac-medium` — good quality, moderate size (~65MB).

Config in `~/.janus/voice_config.json`:
```json
{
  "backend": "piper",
  "piper_binary": "~/.janus/bin/piper",
  "voice_model": "~/.janus/voices/en_US-lessac-medium.onnx",
  "playback": "termux-media-player"
}
```

## Data flow

```
JANUS response (text)
    |
    v
print to terminal (always)
    |
    v
voice_enabled()? ── no ──> done
    |
    yes
    v
voice.speak(text)
    |
    v
chunk into sentences
    |
    v
for each chunk:
    piper --model X --output_raw | playback
    (background thread, non-blocking)
```

## What this does NOT include (phase 2)

- Speech-to-text / voice input
- Wake-word detection ("hey" / "send")
- Per-message voice granularity (all or nothing for now)
- Voice selection UI (just config file)
- Streaming TTS from partial LLM output (waits for full response)

## File inventory

| File | New/Modified | Purpose |
|------|-------------|---------|
| `janus/voice.py` | New | TTS engine abstraction + Piper backend |
| `janus/voice_toggle.sh` | New | Notification toggle script |
| `janus/shell.py` | Modified | Integration points (init, post-LLM, post-narrate, !voice cmd) |
| `~/.janus/voice_config.json` | New (runtime) | TTS configuration |
| `~/.janus/voice_enabled` | New (runtime) | Toggle state file |
| `~/.janus/bin/piper` | New (runtime) | Piper binary |
| `~/.janus/voices/` | New (runtime) | Voice model files |

## Open questions

1. **Piper in proot:** Piper's aarch64 binary should run in proot, but we need to verify it doesn't hit dynamic linker issues. Fallback: build from source or use the Python piper-tts package.
2. **Termux:API from proot:** Need to test the exact path and whether notifications work when called from inside the proot container. May need a wrapper script on the host side.
3. **Audio latency:** Sentence chunking adds latency on the first chunk (Piper must render before playback starts). For short responses this is fine. For long responses the streaming chunking approach should keep it feeling responsive. Need to measure actual latency on the Pixel 7 Pro.
