#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select salt-stratified Top-N 5C predictions with exact seqA deduplication.

The input is the full prediction table produced by predict_5c.py.
For each salt system, exact duplicate seqA strings are collapsed and the
highest-scoring row is retained. The unique rows are ranked independently
within each salt system and capped at Top-N.

Processing is chunked. The retained Top-N rows from earlier chunks are merged
with each new chunk before deduplication and ranking, which preserves the
global Top-N unique sequences for each group without loading the full table.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

SCORE_CANDIDATES = ("pred_retention_5C", "pred_retention", "score", "prediction")
SEQ_CANDIDATES = ("seqA", "seqA_molpercent", "SMITED_input", "smiles")

# LiPO2F2 and LiDFP denote the same auxiliary salt in this project.
SALT_ALIASES = {
    "lipf6": "LiPF6",
    "lifsi": "LiFSI",
    "litfsi": "LiTFSI",
    "lidfob": "LiDFOB",
    "lidfp": "LiDFP",
    "lipo2f2": "LiDFP",
}


def canonical_salt(value: object) -> str:
    if pd.isna(value):
        return "O"
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "null", "o"}:
        return "O"
    key = re.sub(r"[\s_+\-]", "", raw).lower()
    return SALT_ALIASES.get(key, raw)


def make_salt_group(row: pd.Series, salt1_col: str, salt2_col: str) -> str:
    salts = [
        canonical_salt(row[salt1_col]),
        canonical_salt(row[salt2_col]),
    ]
    salts = sorted(salt for salt in salts if salt != "O")
    return "+".join(salts) if salts else "UNKNOWN"


def choose_column(columns, requested, candidates, label: str) -> str:
    if requested:
        if requested not in columns:
            raise ValueError(
                f"{label} column '{requested}' not found. Available: {list(columns)}"
            )
        return requested
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"Cannot identify {label} column. Expected one of {candidates}; "
        f"available columns: {list(columns)}"
    )


def safe_filename(value: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", str(value))


def keep_top_n(
    frame: pd.DataFrame, score_col: str, seq_col: str, top_n: int
) -> pd.DataFrame:
    """Exact seqA deduplication; retain highest score, then rank and cap."""
    if frame.empty:
        return frame
    ranked = frame.sort_values(score_col, ascending=False, kind="mergesort")
    unique = ranked.drop_duplicates(subset=[seq_col], keep="first")
    return unique.head(top_n).copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Full 5C prediction CSV")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument("--top-n", type=int, default=100_000)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--score-col", default=None)
    parser.add_argument("--seq-col", default=None)
    parser.add_argument("--salt1-col", default="salt1")
    parser.add_argument("--salt2-col", default="salt2")
    parser.add_argument("--condition-tag", default="5C")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if args.top_n < 1 or args.chunk_size < 1:
        raise ValueError("--top-n and --chunk-size must be positive")

    columns = pd.read_csv(input_csv, nrows=0).columns
    score_col = choose_column(columns, args.score_col, SCORE_CANDIDATES, "score")
    seq_col = choose_column(columns, args.seq_col, SEQ_CANDIDATES, "sequence")
    for salt_col in (args.salt1_col, args.salt2_col):
        if salt_col not in columns:
            raise ValueError(
                f"Salt column '{salt_col}' is missing; cannot stratify candidates."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start = time.time()

    retained: dict[str, pd.DataFrame] = {}
    total_rows = valid_rows = dropped_scores = dropped_sequences = chunks = 0

    for chunk in pd.read_csv(input_csv, chunksize=args.chunk_size):
        chunks += 1
        total_rows += len(chunk)

        chunk[score_col] = pd.to_numeric(chunk[score_col], errors="coerce")
        before = len(chunk)
        chunk = chunk.dropna(subset=[score_col]).copy()
        dropped_scores += before - len(chunk)
        if chunk.empty:
            continue

        chunk[seq_col] = chunk[seq_col].fillna("").astype(str).str.strip()
        before = len(chunk)
        chunk = chunk.loc[chunk[seq_col].ne("")].copy()
        dropped_sequences += before - len(chunk)
        valid_rows += len(chunk)
        if chunk.empty:
            continue

        chunk["salt_group"] = chunk.apply(
            lambda row: make_salt_group(row, args.salt1_col, args.salt2_col),
            axis=1,
        )

        # Process each salt system independently. This avoids cross-group
        # collisions and keeps the sequence uniqueness rule explicit.
        for group, group_chunk in chunk.groupby("salt_group", sort=False):
            if group in retained:
                combined = pd.concat([retained[group], group_chunk], ignore_index=True)
            else:
                combined = group_chunk
            retained[group] = keep_top_n(combined, score_col, seq_col, args.top_n)

    if not retained:
        raise RuntimeError("No valid prediction rows remained.")

    summary_rows = []
    exported_total = 0
    group_order = sorted(
        retained,
        key=lambda group: float(retained[group][score_col].max()),
        reverse=True,
    )

    for group in group_order:
        result = keep_top_n(
            retained[group], score_col, seq_col, args.top_n
        ).reset_index(drop=True)
        result.insert(0, "rank_in_salt_group", range(1, len(result) + 1))

        filename = (
            f"Top{args.top_n}_{safe_filename(args.condition_tag)}_"
            f"{safe_filename(group)}_seqA_dedup_{timestamp}.csv"
        )
        result.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")
        exported_total += len(result)

        summary_rows.append({
            "salt_group": group,
            "export_unique_seqA_rows": len(result),
            "max_score": float(result[score_col].max()),
            "mean_score": float(result[score_col].mean()),
            "min_score": float(result[score_col].min()),
            "output_file": filename,
        })

    summary_path = (
        output_dir
        / f"summary_stats_{args.condition_tag}_seqA_dedup_{timestamp}.csv"
    )
    pd.DataFrame(summary_rows).to_csv(
        summary_path, index=False, encoding="utf-8-sig"
    )

    run_info = {
        "task": "salt_stratified_top_n_exact_seqA_dedup",
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "condition_tag": args.condition_tag,
        "timestamp": timestamp,
        "score_column": score_col,
        "sequence_column": seq_col,
        "salt_columns": [args.salt1_col, args.salt2_col],
        "top_n_per_salt_group": args.top_n,
        "chunk_size": args.chunk_size,
        "total_rows_read": total_rows,
        "valid_rows": valid_rows,
        "dropped_invalid_scores": dropped_scores,
        "dropped_empty_sequences": dropped_sequences,
        "chunk_count": chunks,
        "salt_group_count": len(retained),
        "export_total_unique_seqA_rows": exported_total,
        "elapsed_seconds": round(time.time() - start, 3),
        "salt_aliases": SALT_ALIASES,
    }
    run_info_path = (
        output_dir
        / f"run_info_{args.condition_tag}_seqA_dedup_{timestamp}.json"
    )
    run_info_path.write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Rows read: {total_rows:,}")
    print(f"Valid rows: {valid_rows:,}")
    print(f"Invalid scores dropped: {dropped_scores:,}")
    print(f"Empty sequences dropped: {dropped_sequences:,}")
    print(f"Salt groups: {len(retained)}")
    print(f"Exported unique seqA rows: {exported_total:,}")
    print(f"Summary: {summary_path}")
    print(f"Run info: {run_info_path}")
    print(f"Elapsed: {run_info['elapsed_seconds']:.2f} s")


if __name__ == "__main__":
    main()
