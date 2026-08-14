from opencood.tools.colif.audit_source_evidence import (
    classify_receiver_support,
    uniformly_sample_keys,
)


def test_classify_receiver_support() -> None:
    assert classify_receiver_support("0", set()) == "unsupported"
    assert classify_receiver_support("0", {"0"}) == "receiver_only"
    assert classify_receiver_support("0", {"1"}) == "partner_only"
    assert classify_receiver_support("0", {"0", "1"}) == "shared"


def test_uniform_sampling_covers_both_sequence_ends() -> None:
    keys = tuple("%06d" % index for index in range(10))
    assert uniformly_sample_keys(keys, 4) == (
        "000000", "000003", "000006", "000009"
    )
    assert uniformly_sample_keys(keys, 1) == ("000005",)
    assert uniformly_sample_keys(keys, None) == keys
