import pytest

from opencood.tools import eval_colif_v2v4real, inference_colif_v2v4real, train_colif
from opencood.tools.colif.receiver_protocol import (
    apply_receiver_override,
    parse_receiver_ids,
)


def test_receiver_override_retains_cmp_requested_ego_order() -> None:
    hypes = {"fusion": {"args": {"receiver_ids": [0]}}}

    assert apply_receiver_override(hypes, "1,0") == (1, 0)
    assert hypes["fusion"]["args"]["receiver_ids"] == [1, 0]


def test_absent_override_preserves_configured_single_receiver() -> None:
    hypes = {"fusion": {"args": {"receiver_ids": [0]}}}

    assert apply_receiver_override(hypes, None) == (0,)
    assert hypes["fusion"]["args"]["receiver_ids"] == [0]


@pytest.mark.parametrize("raw", ("", "0,,1", "0,0", "-1"))
def test_receiver_protocol_rejects_invalid_receiver_lists(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_receiver_ids(raw)


def test_checkpoint_hypes_can_be_explicitly_evaluated_for_both_receivers() -> None:
    hypes = {"fusion": {"args": {"receiver_ids": [0]}}}

    assert apply_receiver_override(hypes, "0,1") == (0, 1)
    assert hypes["fusion"]["args"]["receiver_ids"] == [0, 1]


@pytest.mark.parametrize(
    "argument_parser, argv",
    (
        (
            train_colif._arguments,
            ["train_colif", "--hypes_yaml", "colif.yaml", "--receiver-ids", "0,1"],
        ),
        (
            eval_colif_v2v4real._arguments,
            [
                "eval_colif_v2v4real",
                "--hypes_yaml",
                "colif.yaml",
                "--checkpoint",
                "checkpoint.pth",
                "--output",
                "metrics.json",
                "--receiver-ids",
                "0,1",
            ],
        ),
        (
            inference_colif_v2v4real._arguments,
            [
                "inference_colif_v2v4real",
                "--hypes_yaml",
                "colif.yaml",
                "--checkpoint",
                "checkpoint.pth",
                "--output",
                "runtime.pth",
                "--receiver-ids",
                "0,1",
            ],
        ),
    ),
)
def test_all_entrypoints_accept_receiver_protocol(
    monkeypatch, argument_parser, argv
) -> None:
    monkeypatch.setattr("sys.argv", argv)

    assert argument_parser().receiver_ids == "0,1"
