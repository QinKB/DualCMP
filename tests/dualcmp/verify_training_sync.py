"""Fail closed unless local and lab-lan have identical frozen acceptance code."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SCOPE = ("opencood", "MTR", "AB3Dmot", "tests", "environment.yml", "setup.py")


def snapshot():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip()
    paths = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--"]
        + list(SCOPE), cwd=str(ROOT)).split(b"\0")
    digest = hashlib.sha256()
    for path in sorted(set(paths) - {b""}):
        value = hashlib.sha256((ROOT / path.decode()).read_bytes()).hexdigest()
        digest.update(value.encode() + b"  " + path + b"\n")
    return {"head": head, "source_manifest_sha256": digest.hexdigest(), "scope": list(SCOPE)}


def verify_proof(proof, current):
    if (proof.get("contract") != "dualcmp_two_host_frozen_source_v1"
            or proof.get("local") != proof.get("remote")
            or proof.get("local") != current):
        raise RuntimeError("acceptance refused: two-host sync proof missing, stale or inconsistent")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--snapshot", action="store_true")
    action.add_argument("--output")
    action.add_argument("--check-proof")
    parser.add_argument("--remote", default="lab-lan")
    parser.add_argument("--remote-project", default=str(ROOT))
    args = parser.parse_args()
    first = snapshot()
    if args.snapshot:
        print(json.dumps(first))
        return
    if args.check_proof:
        verify_proof(json.loads(Path(args.check_proof).read_text()), first)
        print(json.dumps({"status": "synchronized", **first}))
        return
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    command = "cd {} && python3 tests/dualcmp/verify_training_sync.py --snapshot".format(
        shlex.quote(args.remote_project))
    remote = json.loads(subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", args.remote, command],
        text=True, timeout=45))
    last = snapshot()
    if first != remote or first != last:
        raise RuntimeError("acceptance refused: code differs across hosts or changed during sync check")
    proof = {"contract": "dualcmp_two_host_frozen_source_v1", "remote_host": args.remote,
             "checked_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
             "local": first, "remote": remote}
    with destination.open("x") as stream:
        json.dump(proof, stream, indent=2)
    print(json.dumps({"status": "synchronized", "proof": str(destination), **first}))


if __name__ == "__main__":
    main()
