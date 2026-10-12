"""Event-driven, single-GPU validation of immutable remote checkpoints."""
import argparse
import csv
import ctypes
import fcntl
import json
import math
import os
from pathlib import Path
import select
import shlex
import subprocess
import sys
import tempfile
import uuid

from opencood.tools.dualcmp.checkpoint_handoff import atomic_json, file_sha256
from opencood.tools.dualcmp.cmp_adapted_forecast_metrics import CONTRACT, PRIMARY_METRICS

ROOT = Path(__file__).resolve().parents[3]


class FileEvents:
    """Watch before scanning; kernel events latch while inference is running."""
    def __init__(self, directories):
        libc = ctypes.CDLL(None, use_errno=True)
        self.fd = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1")
        try:
            for directory in directories:
                # CLOSE_WRITE, MOVED_TO, DELETE_SELF and MOVE_SELF.
                if libc.inotify_add_watch(self.fd, os.fsencode(str(directory)), 0x8 | 0x80 | 0x400 | 0x800) < 0:
                    raise OSError(ctypes.get_errno(), "inotify_add_watch", str(directory))
        except Exception:
            self.close()
            raise

    def wait(self, timeout=None):
        if not select.select([self.fd], [], [], timeout)[0]:
            raise TimeoutError("checkpoint transfer made no filesystem progress before deadline")
        # Always reconcile the durable manifests, including after queue overflow.
        while True:
            try:
                os.read(self.fd, 65536)
            except BlockingIOError:
                return

    def close(self):
        os.close(self.fd)


def ready_checkpoint(root, event):
    if event.get("version") != "dualcmp_external_checkpoint_v1":
        raise ValueError("unknown checkpoint event contract")
    name = "checkpoint_epoch_%03d_step_%09d.pt" % (event["epoch"], event["step"])
    if event["checkpoint"] != name or event["forecast_evaluation"] != CONTRACT:
        raise ValueError("checkpoint name or evaluation protocol differs from frozen contract")
    path = root / name
    if path.is_symlink():
        raise ValueError("immutable checkpoint must not be a symlink")
    if not path.is_file() or path.stat().st_size != event["size_bytes"]:
        return None
    before = path.stat()
    digest = file_sha256(path)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        return None
    return path if digest == event["sha256"] else None


def checked_result(path, event, train_window):
    report = json.loads(path.read_text())
    if (report["evaluation_protocol"] != CONTRACT
            or report["original_checkpoint_sha256"] != event["sha256"]
            or report["checkpoint_epoch"] != event["epoch"]
            or report["checkpoint_step"] != event["step"]
            or report["full_test"] == train_window
            or not report["external_train_bn_calibration"]["checkpoint_unchanged"]):
        raise RuntimeError("validation result does not describe the announced checkpoint/scope")
    metrics = {key: float(report["model"][key]) for key in PRIMARY_METRICS}
    if not all(math.isfinite(value) and value >= 0 for value in metrics.values()):
        raise RuntimeError("non-finite/negative forecast metric")
    if not 0 <= metrics["MR_2.0"] <= 1 or report["dataset_samples"] <= 0 or report["model"]["count"] <= 0:
        raise RuntimeError("invalid metric population")
    with path.with_suffix(".csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1 or any(float(rows[0][key]) != metrics[key] for key in PRIMARY_METRICS):
        raise RuntimeError("primary CSV differs from full result")
    return metrics


def successful_records(output, train_window):
    records = []
    for path in sorted(output.glob("epoch*_step*/success.json")):
        record = json.loads(path.read_text())
        if record["train_window"] != train_window or record["evaluation_protocol"] != CONTRACT:
            raise RuntimeError("cannot mix validation protocols or sample scopes")
        records.append(record)
    return records


def write_summary(output, records):
    fields = ("evaluation_protocol", "checkpoint_epoch", "checkpoint_step") + PRIMARY_METRICS
    with tempfile.NamedTemporaryFile(mode="w", newline="", dir=str(output), delete=False) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda row: row["step"]):
            writer.writerow(dict(evaluation_protocol=CONTRACT["protocol"],
                checkpoint_epoch=record["epoch"], checkpoint_step=record["step"], **record["metrics"]))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(output / "metrics.csv"))


def retain(root, output, records, remote):
    from opencood.tools.dualcmp.checkpoint_retention import prune_checkpoints
    ranked = sorted(records, key=lambda row: (row["metrics"]["minADE_5s"],
        row["metrics"]["MR_2.0"], row["metrics"]["minFDE_5s"], row["step"]))
    plan = {"contract": "dualcmp_evaluated_top3_v1", "primary_metric": "minADE_5s",
            "ranked_keep": [row["checkpoint"] for row in ranked[:3]],
            "evaluated": [{key: row[key] for key in ("checkpoint", "sha256", "step")} for row in records]}
    atomic_json(output / "retention_plan.json", plan)
    if remote:
        command = "cd {} && {} -m opencood.tools.dualcmp.checkpoint_retention --run-root {} --plan-stdin".format(
            shlex.quote(str(ROOT)), shlex.quote(sys.executable), shlex.quote(str(root)))
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", remote, command],
            input=json.dumps(plan), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        if result.returncode:
            raise RuntimeError("remote checkpoint retention failed: " + result.stderr[-2000:])
        atomic_json(output / "remote_retention.json", json.loads(result.stdout))
    # Sync can deliver immutable weights before the latest resume alias.
    latest = root / "checkpoint.pt"
    if latest.is_file():
        latest_sha = file_sha256(latest)
        manifests = [json.loads(path.read_text())
                     for path in (root / "checkpoint_events").glob("checkpoint_step_*.json")]
        if any(event["sha256"] == latest_sha and ready_checkpoint(root, event) is not None
               for event in manifests):
            prune_checkpoints(root, plan)


def verify_sync(path):
    subprocess.run([sys.executable, "tests/dualcmp/verify_training_sync.py", "--check-proof", str(path)],
                   cwd=str(ROOT), check=True, stdout=subprocess.DEVNULL)


def consume(args):
    root, output = Path(args.run_root).resolve(), Path(args.output).resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("training and validation outputs must be separate sibling directories")
    output.mkdir(parents=True, exist_ok=True)
    events_dir = root / "checkpoint_events"
    events_dir.mkdir(parents=True, exist_ok=True)
    with (output / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_sync(args.sync_proof)
        events = FileEvents([root, events_dir])
        try:
            atomic_json(output / "worker_ready.json", {"status": "listening", "run_root": str(root)})
            if args.ready_fifo:
                with open(args.ready_fifo, "w") as stream:
                    stream.write("ready\n")
            while True:
                records = successful_records(output, args.train_window)
                completed = {row["step"]: row for row in records}
                pending = False
                for event_path in sorted(events_dir.glob("checkpoint_step_*.json")):
                    event = json.loads(event_path.read_text())
                    if event["step"] in completed:
                        if completed[event["step"]]["sha256"] != event["sha256"]:
                            raise RuntimeError("completed checkpoint identity changed")
                        continue
                    checkpoint = ready_checkpoint(root, event)
                    if checkpoint is None:
                        pending = True
                        continue
                    verify_sync(args.sync_proof)
                    directory = output / ("epoch%03d_step%09d" % (event["epoch"], event["step"]))
                    attempt = directory / ("attempt_" + uuid.uuid4().hex)
                    attempt.mkdir(parents=True)
                    result = attempt / "result.json"
                    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
                        "tests/dualcmp/eval_cmp_iou05_ddp.py", "--checkpoint", str(checkpoint), "--output", str(result),
                        "--train-window" if args.train_window else "--full-test", "--refresh-train-bn",
                        "--batch-size", str(args.batch_size), "--workers", str(args.workers), "--partner-delay-ms", "100"]
                    with (attempt / "evaluation.log").open("w") as log:
                        status = subprocess.run(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT).returncode
                    if status:
                        atomic_json(attempt / "failure.json", {"exit_code": status, "checkpoint": event})
                        raise RuntimeError("native validation failed; see " + str(attempt / "evaluation.log"))
                    metrics = checked_result(result, event, args.train_window)
                    record = dict(event, metrics=metrics, result=str(result), train_window=args.train_window,
                                  evaluation_protocol=CONTRACT)
                    atomic_json(directory / "success.json", record)
                    records.append(record)
                    completed[event["step"]] = record
                    write_summary(output, records)
                    retain(root, output, records, args.remote)
                finish = root / "training_finished.json"
                if finish.is_file():
                    terminal = json.loads(finish.read_text())
                    expected = terminal["expected_events"]
                    if not isinstance(expected, list) or (not expected and int(terminal["exit_code"]) == 0):
                        raise RuntimeError("successful training must publish at least one checkpoint")
                    if any(not name.startswith("checkpoint_step_") or not name.endswith(".json")
                           or Path(name).name != name for name in expected):
                        raise ValueError("invalid completion event identity")
                    expected_steps = [int(name[len("checkpoint_step_"):-len(".json")]) for name in expected]
                    if all(step in completed for step in expected_steps):
                        retain(root, output, records, args.remote)
                        status = int(terminal["exit_code"])
                        atomic_json(output / "completed.json", {"exit_code": status, "evaluated": len(records),
                            "training": terminal, "top3_metric": "minADE_5s"})
                        failure = output / "failed.json"
                        if failure.is_file():
                            failure.unlink()
                        return status
                    pending = True
                # No periodic filesystem/log/GPU polling. Timeout only incomplete transfer.
                events.wait(1800 if pending else None)
        finally:
            events.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sync-proof", required=True)
    parser.add_argument("--ready-fifo")
    parser.add_argument("--remote", default="")
    parser.add_argument("--train-window", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("invalid batch size or worker count")
    try:
        return consume(args)
    except Exception as error:
        atomic_json(Path(args.output) / "failed.json", {"error": str(error)})
        raise


if __name__ == "__main__":
    sys.exit(main())
