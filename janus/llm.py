"""
JANUS 2 LLM layer — swappable backends.

Backend is selected at startup from config:
  "anthropic"  — Anthropic API (direct HTTP, no SDK dependency)
  "ollama"     — local Ollama instance (original JANUS backend)

The Anthropic backend request format was established empirically through
the Officina project. When that format is recovered, fill in
_anthropic_chat() below. The interface is otherwise complete.

Two components:
  think_fetch()   — Main LLM. Blocking network call, returns raw text.
                    Safe to run under a spinner.
  think_display() — Parses ACTION block, typewriter-prints response.
  agent_fix()     — Terminal Agent. Given failed command + error, returns
                    corrected command. Uses faster model if available.
  summarize()     — Compress old history into a compact summary.
"""

import json
import re
import sys
import socket
import urllib.request
import urllib.error
from janus.config import OLLAMA_HOST, OLLAMA_MODEL, LLM_BACKEND, ANTHROPIC_MODEL, ANTHROPIC_AGENT_MODEL

ACTION_RE = re.compile(r'<ACTION>\s*(\{.*?\})\s*</ACTION>', re.DOTALL)

_AGENT_PROMPT = """You are a terminal command corrector.
A shell command was run and failed. Given the original command and the error output,
return a single corrected shell command that fixes the problem.

Output ONLY the corrected shell command. No explanation, no markdown, no backticks.
If the command cannot be meaningfully corrected, output the original command unchanged.
"""

_SUMMARIZE_PROMPT = """You are a concise summarizer.
Summarize the following conversation into 3-5 sentences capturing:
- Key decisions made
- Commands that ran and what they returned
- Important context established
- Where things stand now

Be factual and specific. No filler.
"""


# ─── Ollama backend ──────────────────────────────────────────────

def _ollama(messages: list[dict], stream: bool = False,
            timeout: int = 120, model: str | None = None) -> str | None:
    """Raw Ollama chat call. Returns full response text or None on error."""
    payload = json.dumps({
        "model": model or OLLAMA_MODEL,
        "messages": messages,
        "stream": stream,
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        if stream:
            tokens = []
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode().strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        tokens.append(token)
                    if chunk.get("done"):
                        break
            return "".join(tokens)
        else:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                return data["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()
        except Exception:
            pass
        raise ConnectionError(f"Ollama HTTP {e.code}: {body or e.reason}")
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as e:
        raise ConnectionError(f"Cannot reach Ollama at {OLLAMA_HOST}: {e}")


# ─── Anthropic backend ───────────────────────────────────────────
#
# The exact request format was established empirically via Officina.
# When that project is recovered, fill in the headers and endpoint here.
# The interface (messages in, text out) is already correct.

def _anthropic_chat(messages: list[dict], system: str | None = None,
                    timeout: int = 120, model: str | None = None) -> str | None:
    """
    Direct HTTP call to Anthropic API (no SDK dependency).

    Format established via Officina — to be filled when recovered.
    Currently raises NotImplementedError so the fallback path activates.
    """
    raise NotImplementedError(
        "Anthropic backend format pending recovery from Officina. "
        "Set LLM_BACKEND=ollama in .env to use Ollama for now."
    )


# ─── Backend dispatcher ──────────────────────────────────────────

def _chat(messages: list[dict], stream: bool = False,
          timeout: int = 120, model: str | None = None) -> str | None:
    """Route to the configured backend."""
    if LLM_BACKEND == "anthropic":
        # Extract system message if present
        system = None
        filtered = []
        for m in messages:
            if m.get("role") == "system":
                system = m.get("content", "")
            else:
                filtered.append(m)
        try:
            return _anthropic_chat(filtered, system=system,
                                   timeout=timeout, model=model)
        except NotImplementedError:
            raise ConnectionError(
                "Anthropic backend not yet configured. "
                "See janus/llm.py _anthropic_chat() or set LLM_BACKEND=ollama."
            )
    else:
        return _ollama(messages, stream=stream, timeout=timeout, model=model)


# ─── Action parser ───────────────────────────────────────────────

def _parse_action(text: str) -> tuple[str, dict | None]:
    """Extract <ACTION>...</ACTION> from response. Returns (clean_text, action|None)."""
    match = ACTION_RE.search(text)
    if not match:
        return text.strip(), None
    try:
        action = json.loads(match.group(1))
        clean = text[:match.start()].strip()
        return clean, action
    except (json.JSONDecodeError, KeyError):
        return text.strip(), None


# ─── Typewriter display ──────────────────────────────────────────

def _print_streaming(text: str, color: str = "\033[96m", name: str = "JANUS"):
    """Print text with a fast typewriter effect."""
    import time
    prefix = f"{color}{name}:\033[0m "
    sys.stdout.write(prefix)
    sys.stdout.flush()
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.008)
    print()


# ─── Public interface ─────────────────────────────────────────────

def think_fetch(user_input: str, history: list[dict],
                system_prompt: str) -> str | None:
    """
    Network phase of think(). Blocks on LLM call, returns raw text.
    No stdout output — safe to run under a spinner.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": user_input},
    ]
    return _chat(messages, stream=(LLM_BACKEND == "ollama"), timeout=120)


def think_display(raw: str, name: str = "JANUS") -> tuple[str, dict | None]:
    """
    Display phase. Parses ACTION block, typewriter-prints clean text.
    Returns (display_text, action_dict | None).
    """
    if not raw:
        return "", None
    clean_text, action = _parse_action(raw)
    if clean_text:
        _print_streaming(clean_text, name=name)
    return clean_text, action


def think(user_input: str, history: list[dict], system_prompt: str,
          name: str = "JANUS") -> tuple[str, dict | None]:
    """Main LLM call. Returns (display_text, action_dict | None)."""
    raw = think_fetch(user_input, history, system_prompt)
    return think_display(raw, name=name)


def agent_fix(command: str, error_output: str) -> str:
    """
    Terminal agent: given a failed command and error, return corrected command.
    Uses the faster agent model when configured.
    """
    messages = [
        {"role": "system", "content": _AGENT_PROMPT},
        {"role": "user", "content": f"Command: {command}\nError:\n{error_output}"},
    ]
    model = ANTHROPIC_AGENT_MODEL if LLM_BACKEND == "anthropic" else None
    try:
        result = _chat(messages, stream=False, timeout=30, model=model)
        return (result or command).strip().strip("`")
    except ConnectionError:
        return command


def summarize(text: str) -> str:
    """Summarize a block of conversation into a compact context entry."""
    messages = [
        {"role": "system", "content": _SUMMARIZE_PROMPT},
        {"role": "user", "content": text},
    ]
    model = ANTHROPIC_AGENT_MODEL if LLM_BACKEND == "anthropic" else None
    try:
        return _chat(messages, stream=False, timeout=60, model=model) or text[:500]
    except ConnectionError:
        return text[:500]
