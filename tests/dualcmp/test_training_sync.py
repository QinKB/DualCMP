import copy
import unittest

from verify_training_sync import verify_proof


class TrainingSyncTest(unittest.TestCase):
    def test_reject_stale_or_one_host_changed(self):
        current = {"head": "a" * 40, "source_manifest_sha256": "b" * 64,
                   "scope": ["opencood", "tests"]}
        proof = {"contract": "dualcmp_two_host_frozen_source_v1",
                 "local": current, "remote": copy.deepcopy(current)}
        verify_proof(proof, current)
        with self.assertRaises(RuntimeError):
            verify_proof(proof, dict(current, head="c" * 40))
        proof["remote"]["source_manifest_sha256"] = "d" * 64
        with self.assertRaises(RuntimeError):
            verify_proof(proof, current)
        with self.assertRaises(RuntimeError):
            verify_proof({}, current)


if __name__ == "__main__":
    unittest.main()
