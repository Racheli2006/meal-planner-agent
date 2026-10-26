"""The tests never talk to the real AI: that would cost money and give different answers
each run. A test that needs an AI answer patches agent.make_client with a fake (see
FakeClient in test_agent.py); one that forgets fails here instead of calling OpenAI."""

import atexit
import shutil
import tempfile
from pathlib import Path

import agent

# Every test's temporary files go in one folder, deleted when the tests end, so none are
# left behind (they once piled up in the project folder by the hundreds).
_TEMP_ROOT = Path(tempfile.mkdtemp(prefix="meal-planner-tests-"))
atexit.register(shutil.rmtree, _TEMP_ROOT, ignore_errors=True)


def temp_dir():
    """A fresh, empty folder for one test's files."""
    return Path(tempfile.mkdtemp(dir=_TEMP_ROOT))


def _no_real_ai():
    raise AssertionError("a test tried to call the real AI: patch agent.make_client with a fake")


agent.make_client = _no_real_ai
