#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_new.smoke import run_smoke


if __name__ == "__main__":
    print(json.dumps(run_smoke(), indent=2))
