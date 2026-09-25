from pathlib import Path

import pytest
import torch

from agent1.registry import ToolRegistry


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


@pytest.fixture
def registry():
    return ToolRegistry.from_json(Path(__file__).resolve().parents[1] / "configs/tool_registry.json")
