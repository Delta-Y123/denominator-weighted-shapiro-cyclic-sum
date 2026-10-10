#!/usr/bin/env python3
"""Independently recompute the objective of a stored RBFGS solution.

The solver writes, for every run, the point it settled on and the objective
value it reports:

    Optimal Solution / 最优解详细信息
        Index i | y_i | x_i = y_i² | Ratio x_i/(p·x_{i+1}+q·x_{i+2})

This tool reads that sheet back, recomputes

    F(x, p) = sum_i x_i / (p*x_{i+1} + q*x_{i+2}),   q = 1 - p,

from the stored ``x_i`` using an independent implementation, and compares the
result with (a) the reported best objective and (b) the sum of the stored
``Ratio`` column.  It is a self-check on the archived numbers: the three values
should agree to the precision the workbook was written with.

Usage::

    python tools/verify_objective.py shapiro_results --n 14 --p 0.5
    python tools/verify_objective.py shapiro_results --all
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SHEETS = {
    "zh": {"summary": "汇总信息", "sol": "最优解详细信息", "runs": "各次运行详情"},
    "en": {"summary": "Summary", "sol": "Optimal Solution", "runs": "Run Details"},
}
P_KEYS = {
    "zh": {"n": "维数 n", "p": "权重 p", "best": "最优目标值"},
    "en": {"n": "Dimension n", "p": "Weight p", "best": "Best Objective"},
}
X_CANDIDATES = ["x_i = y_i²", "x_i = yi²", "x_i"]
NAME_RE = re.compile(r"shapiro_n(\d+)_p([\d.]+?)(?:_batch|_\d{8}_\d{6})?\.xlsx$")


def objective(x: np.ndarray, p: float) -> float:
    """F(x, p) = sum_i x_i / (p x_{i+1} + q x_{i+2}), cyclic, q = 1 - p."""
    q = 1.0 - p
    n = len(x)
    total = 0.0
    for i in range(n):
        denom = p * x[(i + 1) % n] + q * x[(i + 2) % n]
        total += x[i] / denom if denom != 0 else np.inf
    return total


def detect(sheets: list[str]) -> str | None:
    for lang, names in SHEETS.items():
        if names["sol"] in sheets:
            return lang
    return None


def inspect(path: Path) -> dict | None:
    with pd.ExcelFile(path) as xl:
        lang = detect(list(xl.sheet_names))
        if lang is None:
            return None
        names = SHEETS[lang]

        info = {}
        if names["summary"] in xl.sheet_names:
            raw = xl.parse(names["summary"])
            info = dict(zip(raw.iloc[:, 0].astype(str).str.strip(), raw.iloc[:, 1]))

        n = info.get(P_KEYS[lang]["n"])
        p = info.get(P_KEYS[lang]["p"])
        best = info.get(P_KEYS[lang]["best"])
        m = NAME_RE.search(path.name)
        if n is None and m:
            n = int(m.group(1))
        if p is None and m:
            p = float(m.group(2))
        if n is None or p is None:
            return None
        n, p = int(float(n)), float(p)

        if names["sol"] not in xl.sheet_names:
            return None
        sol = xl.parse(names["sol"])

    col_x = next((c for c in sol.columns if str(c).strip() in X_CANDIDATES), None)
    if col_x is None:
        # Fall back to whatever column contains a '²' or the third column.
        col_x = next((c for c in sol.columns if "²" in str(c)), sol.columns[2])
    ratio_col = next((c for c in sol.columns if "atio" in str(c) or "比值" in str(c)), None)

    x = pd.to_numeric(sol[col_x], errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size < n:
        x = pd.to_numeric(sol.iloc[:n, 2], errors="coerce").to_numpy(dtype=float)

    recomputed = objective(x, p)
    ratio_sum = None
    if ratio_col is not None:
        r = pd.to_numeric(sol[ratio_col], errors="coerce").to_numpy(dtype=float)
        ratio_sum = float(np.nansum(r))

    best_f = float(best) if best is not None and str(best) not in {"-", "nan"} else None
    return {
        "file": path.name, "n": n, "p": p, "best_reported": best_f,
        "best_recomputed": recomputed, "ratio_sum": ratio_sum,
        "gap_reported": None if best_f is None else recomputed - best_f,
        "gap_ratio": None if ratio_sum is None else recomputed - ratio_sum,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--n", type=int, action="append")
    ap.add_argument("--p", type=float, action="append")
    ap.add_argument("--all", action="store_true", help="check every workbook")
    args = ap.parse_args(argv)

    rows = []
    for path in sorted(args.results_dir.glob("shapiro_n*_p*.xlsx")):
        if not args.all:
            m = NAME_RE.search(path.name)
            if not m:
                continue
            if args.n and int(m.group(1)) not in args.n:
                continue
            if args.p and not any(abs(float(m.group(2)) - q) < 1e-9 for q in args.p):
                continue
        rec = inspect(path)
        if rec:
            rows.append(rec)

    if not rows:
        print("nothing matched", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows).sort_values(["n", "p", "file"])
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.10g}"))

    bad = df[df["gap_reported"].abs() > 1e-6]
    print(f"\n{len(df)} workbook(s) checked; "
          f"{len(bad)} where the reported objective differs from the "
          f"recomputed one by more than 1e-6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
