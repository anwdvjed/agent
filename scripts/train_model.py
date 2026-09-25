#!/usr/bin/env python3
"""Convenience wrapper for the agent1 CLI."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent1.cli import main
if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
