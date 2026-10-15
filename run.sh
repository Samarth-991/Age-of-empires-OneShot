#!/usr/bin/env bash
# Run the AoE2 Coaching Agent from the project root.
# Usage:
#   ./run.sh web                            # web dashboard on http://127.0.0.1:8000
#   ./run.sh web --port 8080                # web dashboard on another port
#   ./run.sh start                          # interactive civ selection
#   ./run.sh start --civ Teutons --opponent Vietnamese
#   ./run.sh test-capture                   # verify screenshot works
#   ./run.sh test-kb Mongols Knight         # verify knowledge base
#   ./run.sh list-civs                      # list all 50 civs

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Modules import both as top-level (`from config import ...`, needs
# src/aoe2_agent on the path) and as a package (`from aoe2_agent.config
# import ...`, needs src on the path).
cd "$ROOT/src/aoe2_agent"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec python cli.py "$@"
