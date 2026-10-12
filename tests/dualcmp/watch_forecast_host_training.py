"""Block on tmux job events and persist them across Codex interruptions."""
import json
import os
from pathlib import Path
import selectors
import sys


def _latch(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    directory = Path(sys.argv[1])
    selector = selectors.DefaultSelector()
    buffers = {}
    for name in ("startup", "complete"):
        descriptor = os.open(str(directory / name), os.O_RDWR | os.O_NONBLOCK)
        selector.register(descriptor, selectors.EVENT_READ, name)
        buffers[name] = b""
    _latch(directory / "listener_ready.json", {"status": "listening"})
    with (directory / "ready").open("w") as ready:
        ready.write("ready\n")
    started = False
    while True:
        for key, _ in selector.select():
            name = key.data
            buffers[name] += os.read(key.fd, 65536)
            if b"\n" not in buffers[name]:
                continue
            line, buffers[name] = buffers[name].split(b"\n", 1)
            if name == "startup":
                payload = json.loads(line.decode("utf-8"))
                _latch(directory / "startup.json", payload)
                started = True
            else:
                payload = {"status": "completed", "exit_code": int(line)}
                _latch(directory / "completed.json", payload)
            if not (directory / "first_event.json").exists():
                first = payload if started else dict(
                    payload, status="finished_before_startup_confirmation")
                _latch(directory / "first_event.json", first)
                with (directory / "first").open("w") as first_event:
                    first_event.write(json.dumps(first) + "\n")
            if name == "complete":
                return


if __name__ == "__main__":
    main()
