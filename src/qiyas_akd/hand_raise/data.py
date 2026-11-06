"""Loading and validating hand-raise training tables.

The hand-raise dataset was a set of per-video CSVs in per-person long format (one row per
``frame, person``) that were concatenated for training. Two silent data bugs from that work are
guarded against here — both produced plausible-looking but wrong results rather than errors:

1. **Phantom class from a header with trailing whitespace.** Some files had the label header
   ``"hand_raised "``. ``pd.concat`` treats it as a *different* column, so rows from those files had
   ``NaN`` in ``"hand_raised"``; ``LabelEncoder`` then turned ``NaN`` into a third class "2". The
   classification report showed a class 2 with zero support, and the binary scorers broke (lesson
   2). Fix: strip every column name *per file before concatenation*, coalesce columns that collide
   after stripping, and refuse labels that are not exactly 0/1.
2. **Grid search returning NaN for every candidate.** ``GridSearchCV(scoring="recall")`` on that
   3-class target cannot compute a binary recall, so every fold scored ``NaN`` and the "best"
   parameters were simply the first grid entry. Fix: validated binary labels (above), an explicit
   ``pos_label=1`` scorer, and ``error_score="raise"`` (see ``models.grid_search_random_forest``).

A third lesson concerns parsing: the original per-person CSVs stored each keypoint as a string
``"(x,y)"`` and one loader kept only the x coordinate (``split(",")[0]``), silently training on half
the features. ``parse_point_columns`` always returns both coordinates.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..keypoints import KEYPOINT_NAMES

LABEL_COLUMN = "hand_raise"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names; merge columns that become identical after stripping.

    When two columns collide (e.g. ``"hand_raised"`` and ``"hand_raised "`` from two source files),
    they are coalesced row-wise (first non-null wins). Conflicting non-null values raise
    ``ValueError`` — that would mean the same row carries two different labels.
    """
    stripped = [str(c).strip() for c in df.columns]
    if len(set(stripped)) == len(stripped):
        out = df.copy()
        out.columns = stripped
        return out
    merged: dict[str, pd.Series] = {}
    for name, (_, col) in zip(stripped, df.items(), strict=True):
        if name not in merged:
            merged[name] = col.copy()
            continue
        prev = merged[name]
        both = prev.notna() & col.notna()
        if (prev[both].astype(str) != col[both].astype(str)).any():
            raise ValueError(f"columns colliding as {name!r} disagree on some rows")
        merged[name] = prev.combine_first(col)
    return pd.DataFrame(merged, index=df.index)


def validate_binary_labels(labels: Iterable, name: str = "label") -> np.ndarray:
    """Return labels as an int array of 0/1 or raise ``ValueError``.

    Accepts ints, floats equal to 0/1, booleans and the strings ``"0"``/``"1"`` (with whitespace).
    Missing values are rejected with a message pointing at the header bug, because silently encoding
    them is exactly what created the phantom class.
    """
    s = pd.Series(list(labels) if not isinstance(labels, pd.Series) else labels)
    if s.isna().any():
        raise ValueError(
            f"{name}: {int(s.isna().sum())} missing labels — check for a label header with "
            "trailing whitespace in some source files (run clean_columns before concatenating)"
        )
    if s.dtype == object:
        s = s.map(lambda v: int(v) if isinstance(v, (bool, np.bool_)) else v)
        s = s.astype(str).str.strip()
    numeric = pd.to_numeric(s, errors="coerce")
    bad = numeric.isna() | ~numeric.isin([0, 1])
    if bad.any():
        examples = sorted({repr(v) for v in s[bad].tolist()})[:5]
        raise ValueError(f"{name}: labels must be binary 0/1, found {', '.join(examples)}")
    return numeric.to_numpy().astype(int)


def parse_point(value: object) -> tuple[float, float]:
    """Parse ``"(x,y)"`` / ``"(x,y,conf)"`` strings (or numbers / tuples) into ``(x, y)``.

    Missing (NaN, ``None``, bare numbers) or unparseable values become ``(0, 0)``, the pose stage's
    "not detected" encoding.
    """
    if isinstance(value, (tuple, list)):
        return (float(value[0]), float(value[1])) if len(value) >= 2 else (0.0, 0.0)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            return (0.0, 0.0)
        return parse_point(parsed)
    return (0.0, 0.0)


def parse_point_columns(
    df: pd.DataFrame, names: Sequence[str] = KEYPOINT_NAMES
) -> pd.DataFrame:
    """Convert per-keypoint ``"(x,y)"`` columns (``head``, ``neck``, …) into ``kp_{k}_x/y``.

    Both coordinates are kept. Other columns are passed through unchanged.
    """
    out = df.drop(columns=list(names)).copy()
    for k, name in enumerate(names):
        pts = df[name].map(parse_point)
        out[f"kp_{k}_x"] = [p[0] for p in pts]
        out[f"kp_{k}_y"] = [p[1] for p in pts]
    return out


def load_tables(
    paths: Sequence[str | Path],
    label_column: str = LABEL_COLUMN,
    group_column: str = "video",
) -> pd.DataFrame:
    """Read and concatenate per-video CSVs safely.

    * column names are cleaned *per file* (``clean_columns``) before ``pd.concat``;
    * a missing ``group_column`` is filled with the file stem, so grouped splits keep each file
      (= video) on one side;
    * per-keypoint ``"(x,y)"`` columns are parsed into ``kp_{k}_x/y`` when present;
    * labels are validated as binary.
    """
    if not paths:
        raise ValueError("no input files")
    frames = []
    for p in paths:
        p = Path(p)
        df = clean_columns(pd.read_csv(p))
        if group_column not in df.columns:
            df[group_column] = p.stem
        if all(n in df.columns for n in KEYPOINT_NAMES):
            df = parse_point_columns(df)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    if label_column not in data.columns:
        raise KeyError(f"label column {label_column!r} not found; columns: {list(data.columns)}")
    data[label_column] = validate_binary_labels(data[label_column], label_column)
    return data
