"""Publish immutable training checkpoints for external validation."""
import hashlib
import fcntl
import json
import os
from pathlib import Path
import tempfile
from contextlib import contextmanager


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def checkpoint_lock(run_root):
    """Serialize publication and pruning for one run directory."""
    with (Path(run_root) / ".checkpoint.lock").open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def prepare_training_output(output, resume=False, external=False):
    """Create a fresh output, allowing only an empty external-event directory."""
    output = Path(output)
    if resume:
        output.mkdir(parents=True, exist_ok=True)
        return output
    if not external:
        output.mkdir(parents=True, exist_ok=False)
        return output
    if output.is_symlink():
        raise ValueError("external training output must not be a symlink")
    if output.exists():
        if not output.is_dir():
            raise FileExistsError("external training output is not a directory")
        entries = list(output.iterdir())
        if entries:
            if len(entries) != 1 or entries[0].name != "checkpoint_events":
                raise FileExistsError("external training output contains pre-existing artifacts")
            events = entries[0]
            if events.is_symlink() or not events.is_dir() or any(events.iterdir()):
                raise FileExistsError("external checkpoint_events must be an empty real directory")
    else:
        output.mkdir(parents=True)
    events = output / "checkpoint_events"
    if not events.exists():
        events.mkdir()
    return output


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=str(path.parent),
                prefix="." + path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent)
    except Exception:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise


def save_external_checkpoint(checkpoint, output: Path):
    """Save an immutable blob, atomically advance checkpoint.pt, then announce it."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with checkpoint_lock(output):
        return _save_external_checkpoint_locked(checkpoint, output)


def _save_external_checkpoint_locked(checkpoint, output):
    from opencood.tools.dualcmp.train_tracking_ddp import _atomic_save

    epoch, step = int(checkpoint["epoch"]), int(checkpoint["step"])
    if epoch < 0 or step < 0:
        raise ValueError("checkpoint epoch and step must be non-negative")
    name = "checkpoint_epoch_%03d_step_%09d.pt" % (epoch, step)
    blob = output / name
    events = output / "checkpoint_events"
    event = events / ("checkpoint_step_%09d.json" % step)
    if blob.exists() or event.exists():
        raise FileExistsError("external checkpoint or event already exists for step %d" % step)

    provenance = checkpoint.get("provenance") or {}
    with tempfile.TemporaryDirectory(prefix=".checkpoint-handoff-", dir=str(output)) as staging:
        staged = Path(staging) / name
        _atomic_save(checkpoint, staged)
        with staged.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(str(staged), str(blob))
    _fsync_directory(output)

    temporary_link = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=".checkpoint.pt.", suffix=".tmp", dir=str(output))
        os.close(descriptor)
        temporary_link = Path(name)
        temporary_link.unlink()
        os.link(str(blob), str(temporary_link))
        os.replace(str(temporary_link), str(output / "checkpoint.pt"))
        _fsync_directory(output)
    except Exception:
        if temporary_link is not None:
            try:
                temporary_link.unlink()
            except FileNotFoundError:
                pass
        raise

    manifest = {
        "version": "dualcmp_external_checkpoint_v1",
        "checkpoint": blob.name,
        "epoch": epoch,
        "step": step,
        "size_bytes": blob.stat().st_size,
        "sha256": file_sha256(blob),
        "source_tree_sha256": provenance.get("source_tree_sha256"),
        "base_sha": provenance.get("base_sha"),
        "forecast_evaluation": dict(provenance.get("forecast_evaluation") or {}),
        "partial_epoch": bool(provenance.get("partial_epoch", False)),
    }
    events.mkdir(parents=True, exist_ok=True)
    _fsync_directory(output)
    atomic_json(event, manifest)
    return manifest
