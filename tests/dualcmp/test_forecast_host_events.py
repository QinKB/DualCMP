import json
import os
from pathlib import Path
import select
import subprocess
import sys

import pytest


def _read_event(descriptor):
    assert select.select([descriptor], [], [], 10)[0], "missing job event"
    return os.read(descriptor, 65536).decode("utf-8").strip()


@pytest.mark.parametrize("started", [False, True])
def test_completion_is_persisted_after_first_event_is_consumed(tmp_path, started):
    descriptors = {}
    for name in ("startup", "complete", "ready", "first"):
        os.mkfifo(str(tmp_path / name))
        descriptors[name] = os.open(str(tmp_path / name), os.O_RDWR | os.O_NONBLOCK)
    watcher = Path(__file__).with_name("watch_forecast_host_training.py")
    process = subprocess.Popen([sys.executable, str(watcher), str(tmp_path)])
    try:
        assert _read_event(descriptors["ready"]) == "ready"
        if started:
            event = {"status": "first_optimizer_step_complete", "global_batch": 4}
            os.write(descriptors["startup"], (json.dumps(event) + "\n").encode())
            assert json.loads(_read_event(descriptors["first"])) == event
        code = 0 if started else 7
        os.write(descriptors["complete"], (str(code) + "\n").encode())
        if not started:
            first = json.loads(_read_event(descriptors["first"]))
            assert first == {"status": "finished_before_startup_confirmation", "exit_code": 7}
        assert process.wait(timeout=10) == 0
        completed = json.loads((tmp_path / "completed.json").read_text())
        assert completed == {"status": "completed", "exit_code": code}
        # Recovery reads durable records, never re-consumes a spent FIFO.
        assert json.loads((tmp_path / "first_event.json").read_text())["status"] == (
            "first_optimizer_step_complete" if started else "finished_before_startup_confirmation")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for descriptor in descriptors.values():
            os.close(descriptor)
