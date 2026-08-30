import os
import shutil
import sys

SCRATCH = "/tmp/builder-tests"
os.makedirs(SCRATCH, exist_ok=True)
os.environ["DASHBOARD_CUSTOM_MODELS"] = os.path.join(SCRATCH, "custom_models.json")
os.environ["OPENCODE_CONFIG_PATH"] = os.path.join(SCRATCH, "opencode_config.json")
os.environ["DASHBOARD_MODELS_LOCAL"] = os.path.join(SCRATCH, "models_local.json")

sys.path.insert(0, "/mnt/raid1_nvme/JanusPro7b")

import pytest
from fastapi.testclient import TestClient

import dashboard  # noqa: E402  (import AFTER env vars are set)

REAL_OPENCODE = "/root/.config/opencode/config.json"


@pytest.fixture
def client():
    return TestClient(dashboard.app)


@pytest.fixture
def scratch():
    """Fresh scratch files + pristine MODELS/custom registry for one test."""
    paths = (os.environ["DASHBOARD_CUSTOM_MODELS"], os.environ["OPENCODE_CONFIG_PATH"])
    for p in paths:
        if os.path.exists(p):
            os.remove(p)
    models_backup = dict(dashboard.MODELS)
    ids_backup = set(getattr(dashboard, "CUSTOM_IDS", set()))
    yield paths
    dashboard.MODELS.clear()
    dashboard.MODELS.update(models_backup)
    dashboard.CUSTOM_IDS = ids_backup
    for p in paths:
        if os.path.exists(p):
            os.remove(p)


@pytest.fixture
def real_opencode_copy():
    """Copy the real opencode config into the scratch path (shape-faithful)."""
    dst = os.environ["OPENCODE_CONFIG_PATH"]
    shutil.copy(REAL_OPENCODE, dst)
    return dst
