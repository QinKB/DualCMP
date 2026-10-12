"""Start local validation first, then remote Source training; no status polling."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from opencood.tools.dualcmp.watch_checkpoints import FileEvents, ROOT


def run(command):
    result = subprocess.run(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError("launch failed: " + result.stdout[-4000:])
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sync-proof", required=True)
    parser.add_argument("--mode", choices=("preflight", "full"), required=True)
    parser.add_argument("--remote", default="lab-lan")
    parser.add_argument("--validation-batch-size", type=int, default=2)
    args = parser.parse_args()
    root, output, proof = (Path(value).resolve() for value in
                           (args.run_root, args.output, args.sync_proof))
    if root.exists() or output.exists():
        raise FileExistsError("paired launch requires new training and validation directories")
    run([sys.executable, "tests/dualcmp/verify_training_sync.py", "--check-proof", str(proof)])
    root.mkdir(parents=True)
    (root / "checkpoint_events").mkdir()
    run(["ssh", args.remote, "mkdir -p -- " + shlex.quote(str(root / "checkpoint_events"))])
    # The proof is immutable, but it is not part of the code snapshot scope.
    run(["scp", str(proof), args.remote + ":" + str(proof)])
    local_launch = run(["bash", "tests/dualcmp/launch_checkpoint_validator.sh",
        "dualcmp-val-" + args.tag, str(root), str(output), str(proof), args.mode, args.remote,
        str(args.validation_batch_size)])
    completion = FileEvents([output])
    try:
        command = "cd {} && bash tests/dualcmp/launch_external_forecast.sh {} {} {} {} {}".format(
            shlex.quote(str(ROOT)), shlex.quote("dualcmp-train-" + args.tag),
            shlex.quote(str(root)), 4, shlex.quote(args.mode), shlex.quote(str(proof)))
        remote_launch = run(["ssh", "-o", "BatchMode=yes", args.remote, command])
        startup = json.loads(remote_launch.strip().splitlines()[-1])
        result = {"local_launcher": local_launch, "remote_launcher": remote_launch,
                  "startup": startup, "run_root": str(root), "validation_output": str(output)}
        if args.mode == "preflight":
            while not (output / "completed.json").is_file():
                if (output / "failed.json").is_file():
                    raise RuntimeError((output / "failed.json").read_text())
                completion.wait()
            result["completion"] = json.loads((output / "completed.json").read_text())
            if result["completion"]["exit_code"] or result["completion"]["evaluated"] != 3:
                raise RuntimeError("three-checkpoint native lifecycle acceptance failed")
        print(json.dumps(result), flush=True)
    finally:
        completion.close()


if __name__ == "__main__":
    main()
