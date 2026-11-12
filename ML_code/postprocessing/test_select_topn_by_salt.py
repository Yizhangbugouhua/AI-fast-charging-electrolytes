import pandas as pd

from select_topn_by_salt import canonical_salt, keep_top_n, make_salt_group


def test_lipo2f2_and_lidfp_are_same_auxiliary_salt():
    assert canonical_salt("LiPO2F2") == "LiDFP"
    assert canonical_salt("LiDFP") == "LiDFP"


def test_dual_salt_group_is_order_independent():
    first = pd.Series({"salt1": "LiPF6", "salt2": "LiFSI"})
    second = pd.Series({"salt1": "LiFSI", "salt2": "LiPF6"})
    assert make_salt_group(first, "salt1", "salt2") == make_salt_group(
        second, "salt1", "salt2"
    )


def test_duplicate_seq_keeps_highest_score_before_top_n():
    frame = pd.DataFrame({
        "seqA": ["A", "A", "B", "C"],
        "pred_retention_5C": [0.70, 0.90, 0.80, 0.60],
    })
    selected = keep_top_n(frame, "pred_retention_5C", "seqA", 2)
    assert selected["seqA"].tolist() == ["A", "B"]
    assert selected.loc[selected["seqA"] == "A", "pred_retention_5C"].iloc[0] == 0.90


def test_identical_sequence_in_different_salt_groups_is_not_cross_deduplicated():
    row_a = pd.Series({"salt1": "LiPF6", "salt2": "O"})
    row_b = pd.Series({"salt1": "LiPF6", "salt2": "LiDFP"})
    assert make_salt_group(row_a, "salt1", "salt2") != make_salt_group(
        row_b, "salt1", "salt2"
    )
