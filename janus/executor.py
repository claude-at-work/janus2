import subprocess
import os
from dataclasses import dataclass


@dataclass
class CommandResult:
    command: str
    stdout: str
    stderr: str
    exit_code: int


BLOCKED_PATTERNS = [
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=",
    "> /dev/",
    ":(){ :|:& };:",
]


def is_blocked(command: str) -> bool:
    cmd_lower = command.lower().strip()
    return any(pattern in cmd_lower for pattern in BLOCKED_PATTERNS)


def execute(command: str, sandbox_dir: str) -> CommandResult:
    """Execute a shell command inside the sandbox directory."""
    os.makedirs(sandbox_dir, exist_ok=True)

    if is_blocked(command):
        return CommandResult(
            command=command,
            stdout="",
            stderr="JANUS blocked this command for safety reasons.",
            exit_code=1,
        )

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=sandbox_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return CommandResult(
            command=command,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(
            command=command,
            stdout="",
            stderr="Command timed out after 30 seconds.",
            exit_code=1,
        )
    except Exception as e:
        return CommandResult(
            command=command,
            stdout="",
            stderr=str(e),
            exit_code=1,
        )
