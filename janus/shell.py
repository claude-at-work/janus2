"""
JANUS shell -- two-component REPL.

Main LLM:       always conversational, signals actions via <ACTION> blocks.
Terminal Agent: executes approved actions, self-corrects once on failure.
"""

import os
import re
import sys
import time
import datetime
import threading

from janus.config import SANDBOX_DIR, OLLAMA_MODEL
from janus.llm import think, think_fetch, think_display, agent_fix, summarize
from janus.executor import execute, is_blocked
from janus.memory import load_context, save_context, export_session, recall_as_context, conduct_recall, maybe_compress, narrate, narrate_context
from janus.personalities import load_personality
from janus.scratch import ScratchPad, note, list_notes, complete_note, clear_notes, format_scratch_context, format_skills_context
from janus.voice import voice_enabled, set_voice_enabled, load_voice_config, PiperTTS

BANNER = r"""
     ██╗ █████╗ ███╗   ██╗██╗   ██╗███████╗
     ██║██╔══██╗████╗  ██║██║   ██║██╔════╝
     ██║███████║██╔██╗ ██║██║   ██║███████╗
██   ██║██╔══██║██║╚██╗██║██║   ██║╚════██║
╚█████╔╝██║  ██║██║ ╚████║╚██████╔╝███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚══════╝
  Just Another Natural Ultimate System  v0.2.0
  Personality: {personality}  |  Model: {model}
"""

HELP_TEXT = """
Commands:
  Type naturally     - just talk, JANUS handles the rest
  !raw <cmd>         - execute a shell command directly, no LLM
  !personality       - switch personality (restart to apply)
  !sandbox           - show current sandbox path
  !memory            - show how many turns are in memory
  !note <content>    - add a note to scratch pad (temporary context)
  !notes             - list all scratch pad notes
  !note <id> ✓      - mark note as complete (removes from context)
  !narrate <query>   - walk the memory graph (no LLM, pure traversal)
  !narrate on        - narrate mode: all input goes through the graph only
  !narrate off       - normal mode: LLM responds (graph pre-context always on)
  !voice             - show voice output status
  !voice on          - enable voice output (TTS)
  !voice off         - disable voice output
  !telepad           - remote GPU status
  !telepad connect   - open gate to remote machine
  !telepad ship      - send training package
  !telepad train     - launch training on remote
  !telepad status    - check training progress
  !telepad pull      - retrieve weights + cache
  !telepad disconnect - close the gate
  !help              - this
  !exit / !quit      - exit

Input:
  Multi-line mode    - paste freely; blank line sends, Ctrl-D sends,
                       Ctrl-C cancels current input
"""

C_CYAN    = "\033[96m"
C_YELLOW  = "\033[93m"
C_GREEN   = "\033[92m"
C_RED     = "\033[91m"
C_MAGENTA = "\033[95m"
C_DIM     = "\033[2m"
C_BOLD    = "\033[1m"
C_RESET   = "\033[0m"

# ─── Traversal spinner ──────────────────────────────────────────
#
# While the LLM thinks, JANUS walks.  The spinner cycles through
# link types at phi-scaled intervals — the same golden spiral that
# governs the aesthetic loss.  Each frame is a node in a tiny
# orbit: analogy → contrast → extension → resolution → metaphor →
# symbol → context → …

PHI = 1.6180339887

_WALK_GLYPHS = [
    ("∿",  "analogy"),
    ("⊥",  "contrast"),
    ("→",  "extension"),
    ("◎",  "resolution"),
    ("≈",  "metaphor"),
    ("✦",  "symbol"),
    ("⊃",  "context"),
]

_TRAIL_CHARS = ["─", "─", "╌", "┄", "┈", " "]


class GraphSpinner:
    """Animated traversal indicator that runs in a background thread."""

    def __init__(self, label: str = "traversing"):
        self._label = label
        self._stop  = threading.Event()
        self._thread: threading.Thread | None = None

    # ── context manager ──
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        # Clear the spinner line
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    # ── animation loop ──
    def _run(self):
        idx = 0
        tick = 0
        base_dt = 0.12          # fastest frame
        while not self._stop.is_set():
            glyph, name = _WALK_GLYPHS[idx % len(_WALK_GLYPHS)]

            # Build a little trailing wake behind the active glyph
            trail_len = (tick % 6) + 2
            trail = ""
            for t in range(trail_len):
                c = _TRAIL_CHARS[min(t, len(_TRAIL_CHARS) - 1)]
                trail += c

            # Peek at next node
            next_glyph, next_name = _WALK_GLYPHS[(idx + 1) % len(_WALK_GLYPHS)]

            line = (
                f"  {C_DIM}{self._label}  "
                f"{C_MAGENTA}{glyph} {C_CYAN}{name}"
                f" {C_DIM}{trail} "
                f"{C_MAGENTA}{next_glyph}{C_RESET}"
            )

            sys.stdout.write(f"\r\033[K{line}")
            sys.stdout.flush()

            # Phi-modulated timing: slow down on odd beats, speed up on even
            phi_factor = PHI if (tick % 3 == 0) else (1.0 / PHI)
            dt = base_dt * phi_factor
            self._stop.wait(dt)

            tick += 1
            if tick % 4 == 0:
                idx += 1


# ─── Action approval gate ─────────────────────────────────────

def approve_action(action: dict, confirm_style: str) -> str | None:
    """
    Show the proposed action to the user and get approval.
    Returns the final command string, or None if rejected.
    """
    command = action.get("command", "").strip()
    reason  = action.get("reason", "")

    if not command:
        return None

    print(f"\n{C_YELLOW}  Action requested:{C_RESET}")
    if reason:
        print(f"  {C_DIM}{reason}{C_RESET}")
    print(f"  {C_CYAN}{command}{C_RESET}\n")

    try:
        choice = input(f"{C_YELLOW}  {confirm_style}{C_RESET}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if choice in ("", "y", "yes"):
        return command
    elif choice.startswith("e"):
        try:
            edited = input(f"{C_YELLOW}  Enter corrected command: {C_RESET}").strip()
            return edited if edited else None
        except (EOFError, KeyboardInterrupt):
            print()
            return None
    else:
        print(f"{C_DIM}  Action cancelled.{C_RESET}")
        return None


# ─── Terminal agent: execute + self-correct once ──────────────

def run_action(command: str, sandbox: str) -> tuple[str, bool]:
    """
    Execute command in sandbox. If it fails, ask the Terminal Agent to fix
    it and retry once. Returns (summary_text, success_bool).
    """
    result = execute(command, sandbox)

    if result.exit_code == 0:
        output = (result.stdout or "").strip()
        return output or "Done.", True

    # First failure -- ask agent to fix
    error_output = (result.stderr or result.stdout or "unknown error").strip()
    print(f"\n{C_DIM}  Command failed. Terminal agent attempting correction...{C_RESET}")

    fixed = agent_fix(command, error_output)

    if fixed and fixed != command and not is_blocked(fixed):
        print(f"  {C_DIM}Retrying: {fixed}{C_RESET}")
        result2 = execute(fixed, sandbox)
        if result2.exit_code == 0:
            output = (result2.stdout or "").strip()
            return output or "Done.", True
        error_output = (result2.stderr or result2.stdout or "").strip()

    return error_output, False


# ─── Personality switcher ─────────────────────────────────────

def switch_personality():
    from janus.personalities import PRESETS, save_personality
    print(f"\n{C_YELLOW}  Choose a personality:{C_RESET}\n")
    keys = list(PRESETS.keys())
    for i, key in enumerate(keys):
        p = PRESETS[key]
        print(f"    {C_CYAN}{i+1}){C_RESET} {C_BOLD}{p['name']}{C_RESET} -- {p['tagline']}")
        print(f"       {C_DIM}{p['description']}{C_RESET}")
    print(f"    {C_CYAN}{len(keys)+1}){C_RESET} {C_BOLD}Custom{C_RESET} -- write your own system prompt")
    print()
    try:
        choice = input(f"{C_YELLOW}  Pick [1-{len(keys)+1}]: {C_RESET}").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if not choice.isdigit():
        return
    idx = int(choice)
    if 1 <= idx <= len(keys):
        selected = keys[idx - 1]
        save_personality(selected)
        print(f"\n  {C_GREEN}Switched to {PRESETS[selected]['name']}.{C_RESET} Restart JANUS to apply.\n")
    elif idx == len(keys) + 1:
        try:
            name = input(f"{C_YELLOW}  Personality name: {C_RESET}").strip() or "Custom"
            print(f"{C_DIM}  Enter system prompt (blank line to finish):{C_RESET}")
            lines = []
            while True:
                line = input("  ")
                if not line:
                    break
                lines.append(line)
            if lines:
                from janus.personalities import _ACTION_INSTRUCTIONS
                save_personality("custom", {
                    "name": name,
                    "tagline": "Your rules.",
                    "description": "Custom personality.",
                    "system": "\n".join(lines) + "\n" + _ACTION_INSTRUCTIONS,
                    "confirm_style": "Run it? [Y/n/edit]: ",
                    "prompt_prefix": "janus> ",
                })
                print(f"\n  {C_GREEN}Saved '{name}'.{C_RESET} Restart JANUS to apply.\n")
        except (EOFError, KeyboardInterrupt):
            print()


# ─── Telepad (remote GPU gate) ────────────────────────────────

def _handle_telepad(user_input: str):
    """Handle !telepad commands. All state stays local."""
    from janus.telepad import (
        load_config, save_config, clear_config, TelepadHost,
        ensure_key, get_public_key, test_connection,
        ship_portable, install_deps, launch_training,
        check_training, retrieve_weights, retrieve_cache,
    )

    parts = user_input.strip().split()
    sub = parts[1] if len(parts) > 1 else ""

    # ── !telepad (status) ──
    if not sub:
        cfg = load_config()
        if cfg:
            print(f"\n{C_DIM}  Telepad gate:{C_RESET} {C_CYAN}{cfg.user}@{cfg.host}:{cfg.port}{C_RESET}")
            result = test_connection(cfg)
            if result.exit_code == 0:
                print(f"  {C_GREEN}Connected.{C_RESET}")
            else:
                print(f"  {C_RED}Unreachable.{C_RESET} {C_DIM}{result.stderr.strip()}{C_RESET}")
        else:
            print(f"\n{C_DIM}  No telepad configured. Use !telepad connect <ip>{C_RESET}")
        print()
        return

    # ── !telepad connect [ip] ──
    if sub == "connect":
        ip = parts[2] if len(parts) > 2 else None
        if not ip:
            try:
                ip = input(f"{C_YELLOW}  Remote IP address: {C_RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
        if not ip:
            print(f"{C_DIM}  No IP provided.{C_RESET}\n")
            return

        try:
            user = input(f"{C_YELLOW}  User [root]: {C_RESET}").strip() or "root"
            port_str = input(f"{C_YELLOW}  Port [22]: {C_RESET}").strip() or "22"
            port = int(port_str)
        except (EOFError, KeyboardInterrupt):
            print()
            return
        except ValueError:
            print(f"{C_RED}  Invalid port.{C_RESET}\n")
            return

        # Ensure SSH key exists
        print(f"\n{C_DIM}  Ensuring SSH key...{C_RESET}")
        key_path = ensure_key()
        pub = get_public_key()

        host = TelepadHost(host=ip, user=user, port=port, key=key_path)

        # Test connection
        print(f"{C_DIM}  Testing connection...{C_RESET}")
        result = test_connection(host)

        if result.exit_code != 0:
            print(f"\n{C_RED}  Connection failed.{C_RESET}")
            print(f"{C_DIM}  {result.stderr.strip()}{C_RESET}")
            if pub:
                print(f"\n{C_DIM}  Public key (paste into remote ~/.ssh/authorized_keys):{C_RESET}")
                print(f"  {C_CYAN}{pub}{C_RESET}")
                print(f"\n{C_DIM}  Or run:{C_RESET}")
                print(f"  {C_CYAN}ssh-copy-id -i {key_path}.pub -p {port} {user}@{ip}{C_RESET}")
            print()
            # Save config anyway so they can retry after adding the key
            save_config(host)
            return

        save_config(host)
        print(f"  {C_GREEN}Gate open.{C_RESET} {C_DIM}{user}@{ip}:{port}{C_RESET}\n")
        return

    # ── !telepad ship ──
    if sub == "ship":
        cfg = load_config()
        if not cfg:
            print(f"{C_RED}  No telepad configured. !telepad connect <ip> first.{C_RESET}\n")
            return

        print(f"{C_DIM}  Packing and shipping portable training package...{C_RESET}")
        result = ship_portable(cfg)
        if result.exit_code == 0:
            print(f"  {C_GREEN}Shipped.{C_RESET}")
            print(f"{C_DIM}  Installing dependencies on remote...{C_RESET}")
            dep_result = install_deps(cfg)
            if dep_result.exit_code == 0:
                print(f"  {C_GREEN}Dependencies installed.{C_RESET}")
            else:
                print(f"  {C_RED}Dep install failed:{C_RESET} {dep_result.stderr.strip()[:200]}")
        else:
            print(f"  {C_RED}Ship failed:{C_RESET} {result.stderr.strip()[:200]}")
        print()
        return

    # ── !telepad train [args] ──
    if sub == "train":
        cfg = load_config()
        if not cfg:
            print(f"{C_RED}  No telepad configured.{C_RESET}\n")
            return

        extra_args = " ".join(parts[2:]) if len(parts) > 2 else ""
        print(f"{C_DIM}  Launching training (detached)...{C_RESET}")
        result = launch_training(cfg, args=extra_args, detached=True)

        if result.exit_code == 0:
            pid = result.stdout.strip()
            print(f"  {C_GREEN}Training launched.{C_RESET} {C_DIM}pid={pid}{C_RESET}")
            print(f"  {C_DIM}Use !telepad status to monitor.{C_RESET}")
        else:
            print(f"  {C_RED}Launch failed:{C_RESET} {result.stderr.strip()[:200]}")
        print()
        return

    # ── !telepad status ──
    if sub == "status":
        cfg = load_config()
        if not cfg:
            print(f"{C_RED}  No telepad configured.{C_RESET}\n")
            return

        result = check_training(cfg)
        if result.exit_code == 0:
            lines = result.stdout.strip().split('\n')
            status = lines[0] if lines else "UNKNOWN"
            color = C_GREEN if status == "RUNNING" else C_CYAN
            print(f"\n  {color}{status}{C_RESET}")
            if len(lines) > 1:
                print(f"{C_DIM}")
                for line in lines[1:]:
                    print(f"  {line}")
                print(f"{C_RESET}")
        else:
            print(f"  {C_RED}Could not reach remote.{C_RESET} {result.stderr.strip()[:200]}")
        print()
        return

    # ── !telepad pull ──
    if sub == "pull":
        cfg = load_config()
        if not cfg:
            print(f"{C_RED}  No telepad configured.{C_RESET}\n")
            return

        print(f"{C_DIM}  Pulling weights...{C_RESET}")
        w_result = retrieve_weights(cfg)
        if w_result.exit_code == 0:
            print(f"  {C_GREEN}Weights retrieved.{C_RESET}")
        else:
            print(f"  {C_RED}Weights pull failed:{C_RESET} {w_result.stderr.strip()[:200]}")

        print(f"{C_DIM}  Pulling embedding cache...{C_RESET}")
        c_result = retrieve_cache(cfg)
        if c_result.exit_code == 0:
            print(f"  {C_GREEN}Cache retrieved.{C_RESET}")
        else:
            print(f"  {C_RED}Cache pull failed:{C_RESET} {c_result.stderr.strip()[:200]}")
        print()
        return

    # ── !telepad disconnect ──
    if sub == "disconnect":
        clear_config()
        print(f"  {C_DIM}Gate closed.{C_RESET}\n")
        return

    print(f"  {C_DIM}Unknown telepad command. Try !help{C_RESET}\n")


# ─── Shutdown ─────────────────────────────────────────────────

def _shutdown(history: list, session_turns: list, session_id: str):
    try:
        save_context(history)
        if session_turns:
            path = export_session(session_turns, session_id)
            if path:
                print(f"\n{C_DIM}  Session saved → {path}{C_RESET}")
    except Exception:
        pass


# ─── Main REPL ───────────────────────────────────────────────

def run():
    personality    = load_personality()
    system_prompt  = personality["system"]
    confirm_style  = personality.get("confirm_style", "Run it? [Y/n/edit]: ")
    prompt_prefix  = personality.get("prompt_prefix", "janus> ")
    persona_name   = personality["name"]

    sandbox = os.path.abspath(SANDBOX_DIR)
    os.makedirs(sandbox, exist_ok=True)

    print(f"{C_CYAN}{BANNER.format(personality=persona_name, model=OLLAMA_MODEL)}{C_RESET}")
    print(f"{C_DIM}  Sandbox: {sandbox}{C_RESET}")

    # ── Load memory ──
    history = load_context()
    if history:
        print(f"{C_DIM}  Resumed {len(history)} turns from last session.{C_RESET}")

    # ── Inject scratch pad notes into history ──
    scratch_context = format_scratch_context()
    if scratch_context:
        history.insert(0, {"role": "system", "content": scratch_context})
        print(f"{C_DIM}  Scratch pad notes loaded.{C_RESET}")

    # ── Inject skills inventory into history ──
    skills_context = format_skills_context()
    if skills_context:
        history.insert(0, {"role": "system", "content": skills_context})
        print(f"{C_DIM}  Skills inventory loaded.{C_RESET}")

    bud_ctx = recall_as_context(f"janus session sandbox:{sandbox}")
    if bud_ctx:
        history.insert(0, {"role": "system", "content": bud_ctx})
        print(f"{C_DIM}  Recalled context from long-term memory.{C_RESET}")

    # Pre-load symlink graph so first query doesn't pay load cost
    from janus.memory import _load_symlink_graph, _graph_chunks
    _load_symlink_graph()
    if _graph_chunks:
        print(f"{C_DIM}  Symlink graph ready ({len(_graph_chunks)} nodes).{C_RESET}")

    # ── Voice output ──
    _voice_config = load_voice_config()
    _voice_engine = PiperTTS(_voice_config)
    if _voice_engine.is_available():
        _voice_status = "ready"
    else:
        _voice_status = "unavailable (piper not found)"
    print(f"{C_DIM}  Voice: {_voice_status} | {'on' if voice_enabled() else 'off'}{C_RESET}")

    print(f"{C_DIM}  Type !help for commands, or just talk.{C_RESET}\n")

    session_id    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_turns = []
    narrate_mode = False

    while True:
        # ── Multi-line input: blank line or Ctrl-D sends, Ctrl-C cancels ──
        # Wait for any ongoing speech to finish before prompting
        if _voice_engine:
            _voice_engine.wait()

        active_prefix = f"{C_MAGENTA}narrate> {C_RESET}" if narrate_mode else f"{C_GREEN}{prompt_prefix}{C_RESET}"
        try:
            first = input(active_prefix)
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C_DIM}Goodbye.{C_RESET}")
            _shutdown(history, session_turns, session_id)
            break

        lines = [first]
        # If the first line is non-empty, collect continuation lines
        if first.strip():
            while True:
                try:
                    cont = input(f"{C_DIM}  ...{C_RESET} ")
                except EOFError:
                    break
                except KeyboardInterrupt:
                    lines = []
                    print()
                    break
                if cont == "":
                    break
                lines.append(cont)

        user_input = "\n".join(lines).strip()
        if not user_input:
            continue

        # ── Special commands ──
        if user_input in ("!exit", "!quit"):
            print(f"{C_DIM}Goodbye.{C_RESET}")
            _shutdown(history, session_turns, session_id)
            break

        if user_input == "!help":
            print(HELP_TEXT)
            continue

        if user_input == "!sandbox":
            print(f"{C_DIM}  Sandbox: {sandbox}{C_RESET}\n")
            continue

        if user_input == "!memory":
            print(f"{C_DIM}  {len(history)} turns in active memory.{C_RESET}\n")
            continue

        if user_input == "!notes":
            notes = list_notes()
            print(f"{C_DIM}  Active scratch pad notes:{C_RESET}")
            if notes:
                for i, note in enumerate(notes):
                    status = "✓" if note.get('checked') else "•"
                    crit = note.get('criticality', 'normal')
                    crit_marker = "!" if crit == 'critical' else "?" if crit == 'important' else ""
                    print(f"    {C_CYAN}{i+1}){C_RESET} {status}{crit_marker} {note.get('content', '')}")
                    if not note.get('checked'):
                        print(f"       {C_DIM}id: {note.get('id')}{C_RESET}")
            else:
                print(f"    {C_DIM}No active notes.{C_RESET}")
            print()
            continue

        if user_input.startswith("!note "):
            note_content = user_input[6:].strip()
            if note_content:
                # Check for completion marker (checkmark symbol)
                if "\u2713" in note_content or "✓" in note_content or note_content.endswith(" complete"):
                    # Mark as complete - extract ID from start of content
                    parts = note_content.replace("\u2713", "").replace("✓", "").replace(" complete", "").strip().split()
                    if len(parts) >= 1 and parts[0].startswith("id:"):
                        note_id = parts[0][3:]
                        success = complete_note(note_id)
                        if success:
                            print(f"{C_GREEN}  Note completed.{C_RESET}\n")
                        else:
                            print(f"{C_RED}  Note not found.{C_RESET}\n")
                    elif len(parts) >= 1:
                        # Assume first word is the ID
                        note_id = parts[0]
                        success = complete_note(note_id)
                        if success:
                            print(f"{C_GREEN}  Note completed.{C_RESET}\n")
                        else:
                            print(f"{C_RED}  Note not found.{C_RESET}\n")
                else:
                    # Add new note
                    result = note(note_content)
                    print(f"{C_GREEN}  Note added: {result.get('content')}{C_RESET}")
                    print(f"    {C_DIM}id: {result.get('id')}{C_RESET}")
                    print()
            continue

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

        if user_input.startswith("!narrate"):
            narrate_arg = user_input[8:].strip()
            if narrate_arg == "on":
                narrate_mode = True
                print(f"{C_MAGENTA}  Narrate mode.{C_RESET} The graph speaks. LLM silent.\n")
                continue
            if narrate_arg == "off":
                narrate_mode = False
                print(f"{C_CYAN}  Normal mode.{C_RESET} LLM responds, graph feeds context.\n")
                continue
            if not narrate_arg:
                status = f"{C_MAGENTA}narrate{C_RESET}" if narrate_mode else f"{C_CYAN}normal{C_RESET}"
                print(f"{C_DIM}  Usage: !narrate <query>  |  !narrate on  |  !narrate off")
                print(f"  Mode: {status}{C_RESET}\n")
                continue
            print()
            with GraphSpinner("walking graph"):
                result = narrate(narrate_arg, depth=5, branches=2)
            if result:
                print(f"{C_MAGENTA}{result}{C_RESET}\n")
                if voice_enabled() and _voice_engine.is_available():
                    _voice_engine.speak(result)
            else:
                print(f"{C_DIM}  The graph doesn't hold that territory yet.{C_RESET}\n")
            continue

        if user_input == "!personality":
            switch_personality()
            continue

        if user_input.startswith("!telepad"):
            _handle_telepad(user_input)
            continue

        if user_input.startswith("!raw "):
            raw_cmd = user_input[5:].strip()
            if raw_cmd:
                if is_blocked(raw_cmd):
                    print(f"{C_RED}  Blocked -- looks dangerous.{C_RESET}\n")
                else:
                    result = execute(raw_cmd, sandbox)
                    if result.stdout:
                        print(result.stdout, end="")
                    if result.stderr:
                        print(f"{C_RED}{result.stderr}{C_RESET}", end="")
                    print()
            continue

        # ── Narrate mode: pure graph traversal, no LLM ──
        if narrate_mode:
            print()
            with GraphSpinner("walking graph"):
                result = narrate(user_input, depth=5, branches=2)
            if result:
                print(f"{C_MAGENTA}{result}{C_RESET}\n")
                if voice_enabled() and _voice_engine.is_available():
                    _voice_engine.speak(result)
            else:
                print(f"{C_DIM}  The graph doesn't hold that territory yet.{C_RESET}\n")
            # Still track the turn so history stays coherent
            history.append({"role": "user", "content": user_input})
            session_turns.append({"role": "user", "content": user_input})
            if result:
                history.append({"role": "assistant", "content": f"[narrate] {result}"})
                session_turns.append({"role": "assistant", "content": f"[narrate] {result}"})
            continue

        # ── Main LLM ──
        try:
            # Shallow recall (1-2 hop graph + FAISS)
            live_ctx = conduct_recall(user_input)
            # Deeper graph walk (3 hop, compact) — pre-context the LLM can cherry-pick
            graph_ctx = narrate_context(user_input)

            parts = [system_prompt]
            if live_ctx:
                parts.append(live_ctx)
            if graph_ctx:
                parts.append(graph_ctx)
            effective_prompt = "\n\n".join(parts)

            # Phase 1: fetch (blocking network call — spinner runs here)
            with GraphSpinner("traversing"):
                raw = think_fetch(user_input, history, effective_prompt)
            # Phase 2: display (typewriter print — no spinner)
            reply, action = think_display(raw, name=persona_name)
        except ConnectionError as e:
            print(f"{C_RED}  Error: {e}{C_RESET}\n")
            continue

        # Speak the reply if voice is enabled
        if voice_enabled() and _voice_engine.is_available():
            _voice_engine.speak(reply)

        # Track conversation turn
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        session_turns.append({"role": "user", "content": user_input})
        session_turns.append({"role": "assistant", "content": reply})

        print()

        # ── Terminal Agent ──
        if action:
            command = action.get("command", "").strip()

            if is_blocked(command):
                print(f"{C_RED}  Action blocked -- looks dangerous.{C_RESET}\n")
                injection = "[Action was blocked for safety reasons.]"
            else:
                approved_cmd = approve_action(action, confirm_style)

                if approved_cmd:
                    with GraphSpinner("executing"):
                        output, success = run_action(approved_cmd, sandbox)

                    status_color = C_GREEN if success else C_RED
                    print(f"{status_color}  {'✓' if success else '✗'} {output[:800]}{C_RESET}\n")

                    injection = f"[command: `{approved_cmd}`]\n[{'output' if success else 'error'}: {output}]"
                else:
                    injection = f"[Action `{command}` was declined by the user.]"

            # Inject result so main LLM knows what happened
            history.append({"role": "user", "content": injection})
            session_turns.append({"role": "user", "content": injection})

        # ── Compress history if getting long ──
        history = maybe_compress(history, summarize)
