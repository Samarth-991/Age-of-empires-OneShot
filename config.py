from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
AOE2_AGENT_ROOT = SRC_ROOT / "aoe2_agent"

LOGS_DIR = PROJECT_ROOT / "logs"
SCREENSHOTS_DIR = LOGS_DIR / "screenshots"
AGENT_LOG_FILE = LOGS_DIR / "agent.log"
CLI_LOG_FILE = LOGS_DIR / "cli.log"
AGENTS_MEMORY_FILE = PROJECT_ROOT / "AGENTS.md"

# Shared screenshots path (WSL)
SCREENSHOT_OUT_DIR: str = str(SCREENSHOTS_DIR)

# Optional override when WSL path mapping differs (e.g. non-/mnt/<drive> layout).
WINDOWS_SCREENSHOT_OUT_DIR = os.environ.get("WINDOWS_SCREENSHOT_OUT_DIR", "")


def _wsl_to_windows_dir(path: str) -> str:
    """Best-effort WSL path (/mnt/e/...) -> Windows path (E:\\...)."""
    p = path.replace("\\", "/")
    if p.startswith("/mnt/") and len(p) > 6:
        drive = p[5]
        rest = p[6:]
        rest_win = rest.replace("/", "\\")
        return f"{drive.upper()}:{rest_win}"
    return p.replace("/", "\\")


SCREENSHOT_OUT_DIR_WINDOWS = (
    WINDOWS_SCREENSHOT_OUT_DIR or _wsl_to_windows_dir(SCREENSHOT_OUT_DIR)
)


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

VLM_MODEL = "qwen2.5vl:7b"
VLM_MAX_OUTPUT_TOKENS = 800  # a full snapshot is ~300 tokens; stops runaway unit lists
RESOURCE_BAR_CROP = (0.5, 0.055)  # (width, height) fraction of the frame holding the top-left resource bar
RESOURCE_BAR_SCALE = 3
OLLAMA_BASE_URL = "http://localhost:11434"

# Strategist model — OpenRouter is preferred when an API key is available.
# If OPENROUTER_API_KEY is set (here or in the environment), the agent uses
# ChatOpenRouter with anthropic/claude-sonnet-4-6.
# Otherwise it falls back to Ollama llama3.1:latest running locally.
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_STRATEGY_MODEL: str = "google/gemma-4-31b-it"
OPENROUTER_VLM_MODEL:str = 'google/gemini-3-flash-preview'

OLLAMA_STRATEGY_MODEL: str = "llama3.1:8b"
OLLAMA_STRATEGY_NUM_CTX = 16384  # deep-agent system prompt + tools + AGENTS.md overflow Ollama's 2048 default
# Fast decision model (Ollama /v1/systemone — requires Ollama 0.35+)
OLLAMA_NIMBLE_MODEL: str = "nimble:latest"
NIMBLE_ENABLED: bool = True
NIMBLE_INTERVAL_SECS: float = 5.0
NIMBLE_TIMEOUT_SECS: float = 120.0
NIMBLE_KEEP_ALIVE: str = "30m"
# Nimble re-reads every question in a request for each answer, so smaller
# batches keep prompts short; the per-cycle question count bounds latency.
NIMBLE_BATCH_SIZE: int = 8
NIMBLE_MAX_QUESTIONS: int = 32
# Legacy alias kept for backward compatibility
STRATEGY_MODEL: str = OLLAMA_STRATEGY_MODEL

# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

KB_PATH = PROJECT_ROOT / "knowledge_base"
KB_CIVS = KB_PATH / "civilizations" / "civs.json"
KB_UNITS = KB_PATH / "units" / "units.json"
KB_COUNTERS = KB_PATH / "counters" / "counters.json"
KB_BUILDINGS = KB_PATH / "buildings" / "building.json"
KB_SUMMARY = KB_PATH / "summary.md"

# ---------------------------------------------------------------------------
# Loop timing (seconds)
# ---------------------------------------------------------------------------

PERCEPTION_INTERVAL_SECS: float = 3.0
STRATEGIST_INTERVAL_SECS: float = 15.0
# Web UI live mode: seconds between automatic screen reads
WEB_LIVE_PERCEPTION_INTERVAL_SECS: float = 10.0

# ---------------------------------------------------------------------------
# Deterministic alarm thresholds (Article V — externally configurable)
# ---------------------------------------------------------------------------

ALARM_FOOD_FLOOR: int = 200
ALARM_WOOD_FLOOR: int = 100
ALARM_GOLD_FLOOR: int = 50
ALARM_IDLE_VILLAGER_SECS: int = 30
ALARM_POP_HEADROOM: int = 5      # alert when pop_cap - population <= this value

# ---------------------------------------------------------------------------
# Screenshot capture
# ---------------------------------------------------------------------------

HASH_THRESHOLD: int = 8   # perceptual-hash diff below which VLM is skipped


# ---------------------------------------------------------------------------
# OpenRouter API key validation
# ---------------------------------------------------------------------------

def check_openrouter_api_key(key: str = "") -> dict:
    """Validate an OpenRouter API key by making a lightweight test call.

    Tries a minimal ChatOpenRouter invocation so the result reflects the actual
    auth state, not just whether the string is non-empty.

    Args:
        key: API key to test. Falls back to OPENROUTER_API_KEY from config / env
             if not provided.

    Returns:
        dict with keys:
            valid   (bool)   — True if the key authenticated successfully
            source  (str)    — where the key came from: "argument", "config", "env", or "none"
            model   (str)    — model that was tested
            message (str)    — human-readable status
    """
    resolved_key = key or OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY", "")
    source = (
        "argument" if key
        else "config" if OPENROUTER_API_KEY
        else "env" if os.environ.get("OPENROUTER_API_KEY")
        else "none"
    )

    if not resolved_key:
        return {
            "valid":   False,
            "source":  source,
            "model":   OPENROUTER_STRATEGY_MODEL,
            "message": (
                "No OPENROUTER_API_KEY found.\n"
                "  • Set it in the environment:  export OPENROUTER_API_KEY=sk-or-...\n"
                "  • Or set it in config.py:     OPENROUTER_API_KEY = 'sk-or-...'\n"
                f"Fallback: Ollama {OLLAMA_STRATEGY_MODEL} will be used instead."
            ),
        }

    os.environ["OPENROUTER_API_KEY"] = resolved_key

    try:
        from langchain_openrouter import ChatOpenRouter
        llm = ChatOpenRouter(
            model=OPENROUTER_STRATEGY_MODEL,
            temperature=0,
            max_tokens=10,
            max_retries=2,
        )
        llm.invoke("ping")
        return {
            "valid":   True,
            "source":  source,
            "model":   OPENROUTER_STRATEGY_MODEL,
            "message": f"OpenRouter API key is valid. Model {OPENROUTER_STRATEGY_MODEL!r} is reachable.",
        }
    except ImportError:
        return {
            "valid":   False,
            "source":  source,
            "model":   OPENROUTER_STRATEGY_MODEL,
            "message": (
                "langchain-openrouter is not installed.\n"
                "  Run:  pip install langchain-openrouter\n"
                f"Fallback: Ollama {OLLAMA_STRATEGY_MODEL} will be used instead."
            ),
        }
    except Exception as exc:
        err = str(exc)
        if "401" in err or "invalid_api_key" in err.lower() or "authentication" in err.lower():
            msg = f"OpenRouter API key is INVALID or expired. ({err[:120]})"
        elif "429" in err or "rate" in err.lower():
            msg = f"OpenRouter rate limit hit — key looks valid but the test was throttled. ({err[:120]})"
        elif "connection" in err.lower() or "timeout" in err.lower():
            msg = f"Network error reaching OpenRouter API. Check your internet connection. ({err[:120]})"
        else:
            msg = f"OpenRouter test call failed: {err[:200]}"
        return {
            "valid":   False,
            "source":  source,
            "model":   OPENROUTER_STRATEGY_MODEL,
            "message": msg,
        }


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  AoE2 Coaching Agent — OpenRouter API Key Check")
    print("=" * 60)

    # Allow passing the key as a CLI argument: python config.py sk-or-...
    cli_key = sys.argv[1] if len(sys.argv) > 1 else ""
    result = check_openrouter_api_key(cli_key)

    status_icon = "✓" if result["valid"] else "✗"
    print(f"\n  {status_icon}  Status  : {'VALID' if result['valid'] else 'NOT VALID'}")
    print(f"     Source  : {result['source']}")
    print(f"     Model   : {result['model']}")
    print(f"\n  {result['message']}")

    if not result["valid"]:
        print(f"\n  Current strategy model will be: Ollama {OLLAMA_STRATEGY_MODEL}")
    else:
        print(f"\n  Current strategy model will be: OpenRouter {result['model']}")

    print()
    sys.exit(0 if result["valid"] else 1)
