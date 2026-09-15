import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from pathlib import Path
import pytest

@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent
