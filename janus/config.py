import os
from pathlib import Path

# ─── LLM backend ─────────────────────────────────────────────────
#
# LLM_BACKEND: "anthropic" | "ollama"
#
# Anthropic: direct HTTP, no SDK. Request format from Officina (pending).
#   ANTHROPIC_MODEL       — main conversational model
#   ANTHROPIC_AGENT_MODEL — faster model for agent_fix/summarize
#
# Ollama: local instance, original JANUS backend.
#   OLLAMA_HOST / JANUS_MODEL as before.

LLM_BACKEND           = os.environ.get("LLM_BACKEND", "ollama")
ANTHROPIC_MODEL       = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
ANTHROPIC_AGENT_MODEL = os.environ.get("ANTHROPIC_AGENT_MODEL", "claude-haiku-4-5-20251001")

OLLAMA_HOST  = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("JANUS_MODEL", "gemma4:31b-cloud")

# ─── Sandbox ─────────────────────────────────────────────────────
# Default to Termux home, not the old Kali chroot path
SANDBOX_DIR = os.environ.get(
    "JANUS_SANDBOX",
    str(Path.home() / "janus2-sandbox")
)

# ─── Personality ─────────────────────────────────────────────────
def _load():
    from janus.personalities import load_personality
    return load_personality()

_personality      = _load()
SYSTEM_PROMPT     = _personality["system"]
PERSONALITY_NAME  = _personality["name"]
CONFIRM_STYLE     = _personality.get("confirm_style", "Run it? [Y/n/edit]: ")
PROMPT_PREFIX     = _personality.get("prompt_prefix", "janus> ")
