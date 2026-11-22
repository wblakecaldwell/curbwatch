"""Settings: environment, then the config file, then the macOS Keychain.

Config and data live outside the source tree so a clone never holds anything personal:
  config file  ~/.config/nestmon/.env          (override with NESTMON_CONFIG)
  data         ~/.local/share/nestmon/         (override with NESTMON_STATE_DIR / NESTMON_DATA_DIR)

Secrets never get printed or logged by this module.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
_XDG_CONFIG = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
_XDG_DATA = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")
CONFIG_FILE = Path(os.environ.get("NESTMON_CONFIG") or _XDG_CONFIG / "nestmon" / ".env").expanduser()
load_dotenv(CONFIG_FILE)

KEYCHAIN_SERVICES = {
    "STARLING_API_KEY": "nestmon-starling",
    "ANTHROPIC_API_KEY": "nestmon-anthropic",
    "LM_API_TOKEN": "nestmon-lmstudio",
}


def _keychain(service: str) -> str | None:
    if not shutil.which("security"):
        return None
    r = subprocess.run(
        ["security", "find-generic-password", "-s", service, "-w"],
        capture_output=True,
        text=True,
    )
    return r.stdout.strip() or None if r.returncode == 0 else None


def get(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name, "").strip()
    if val:
        return val
    if name in KEYCHAIN_SERVICES:
        val = _keychain(KEYCHAIN_SERVICES[name])
        if val:
            return val
    return default


def require(name: str) -> str:
    val = get(name)
    if not val:
        svc = KEYCHAIN_SERVICES.get(name)
        hint = f" or run: security add-generic-password -a \"$USER\" -s {svc} -w" if svc else ""
        raise SystemExit(f"{name} is not set. Put it in {CONFIG_FILE}{hint}")
    return val


def get_float(name: str, default: float) -> float:
    val = get(name)
    return float(val) if val else default


def get_int(name: str, default: int) -> int:
    val = get(name)
    return int(val) if val else default


def hub() -> str:
    """Hub address (IP or hostname). Required; there is no sensible default."""
    return require("STARLING_HUB")


STARLING_HTTPS_PORT = get_int("STARLING_HTTPS_PORT", 3443)
STARLING_HTTP_PORT = get_int("STARLING_HTTP_PORT", 3080)

MODEL = get("CLAUDE_MODEL", "claude-opus-5-5")
# "api" = Anthropic API directly (needs ANTHROPIC_API_KEY);
# "agy" = route through the Antigravity CLI instead.
VISION_BACKEND = get("VISION_BACKEND", "api")
AGY_MODEL = get("AGY_MODEL", "claude-opus-5-5-low")
AGY_BIN = get("AGY_BIN")

# Where `nestmon snap` writes snapshots, and where the curb check keeps its
# crop setup, reference frames and history. Both hold pictures of your property.
DATA_DIR = Path(get("NESTMON_DATA_DIR", str(_XDG_DATA / "nestmon" / "snapshots"))).expanduser()
STATE_DIR = Path(get("NESTMON_STATE_DIR", str(_XDG_DATA / "nestmon" / "curb"))).expanduser()
