#!/usr/bin/env python3
"""Collect the timestamped RBFGS result workbooks into tidy, verified CSV summaries.

``RBFGS_Algorithm.py`` and ``WeightedShapiro.py`` write one workbook per
(n, p, timestamp) under ``shapiro_results/``.  Depending on which revision of
the scripts produced them, the sheets are named in Chinese or in English::

    zh:  汇总信息 / 各次运行详情 / 最优解详细信息 / 统计信息 / 收敛历史
    en:  Summary  / Run Details  / Optimal Solution / Statistics / Convergence History

The script produces two CSVs:

* ``runs_all.csv``     one row per individual optimisation run, as recorded
* ``best_by_case.csv`` one row per (n, p) case

Every value in ``best_by_case.csv`` is *verified*: the point stored in the
workbook's "Optimal Solution" sheet is read back and the objective

    F(x, p) = sum_i x_i / (p*x_{i+1} + q*x_{i+2}),   q = 1 - p,

is recomputed from scratch with an independent implementation.  A workbook
whose own "Best Objective" disagrees with that recomputation is reported as
inconsistent and excluded from the verified minimum, because such a workbook
does not describe a point that actually attains the value it claims.  (One such
file exists in the original archive; see ``n_inconsistent`` below.)

Usage::

    python tools/summarize_results.py shapiro_results -o results/summary
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# --- sheet names, by language -------------------------------------------------
SHEETS = {
    "zh": {"summary": "汇总信息", "runs": "各次运行详情", "sol": "最优解详细信息"},
    "en": {"summary": "Summary", "runs": "Run Details", "sol": "Optimal Solution"},
}

# --- column names, by language ------------------------------------------------
COLS = {
    "zh": {
        "n": "维数 n", "p": "权重 p", "q": "权重 q", "runs": "运行次数",
        "total_iter": "总迭代次数", "total_time": "总计算时间(秒)",
        "run": "运行序号", "iter": "迭代次数", "time": "计算时间(秒)",
        "obj": "最终目标值", "grad": "最终梯度范数", "conv": "是否收敛",
        "osc": "检测到交替模式", "osc_a": "交替模式a值", "osc_b": "交替模式b值",
        "best": "最优目标值",
    },
    "en": {
        "n": "Dimension n", "p": "Weight p", "q": "Weight q", "runs": "Number of Runs",
        "total_iter": "Total Iterations", "total_time": "Total Time (s)",
        "run": "Run Index", "iter": "Iterations", "time": "Time (s)",
        "obj": "Final Objective", "grad": "Final Gradient Norm", "conv": "Converged",
        "osc": "Alternating Pattern Detected", "osc_a": "Alternating a Value",
        "osc_b": "Alternating b Value", "best": "Best Objective",
    },
}

FILENAME_RE = re.compile(
    r"shapiro_n(?P<n>\d+)_p(?P<p>\d+(?:\.\d+)?)(?:_(?P<stamp>\d{8}_\d{6})|_batch)?\.xlsx$"
)

TRUE_TOKENS = {"是", "True", "true", "TRUE", "1", "Y", "yes"}
FALSE_TOKENS = {"否", "False", "false", "FALSE", "0", "N", "no"}

# How far a workbook's own "Best Objective" may sit from the objective
# recomputed at its stored solution before the workbook is called inconsistent.
TOL = 1e-6


def objective(x: np.ndarray, p: float) -> float:
    """F(x, p) = sum_i x_i / (p x_{i+1} + q x_{i+2}), cyclic, q = 1 - p."""
    q = 1.0 - p
    n = len(x)
    total = 0.0
    for i in range(n):
        denom = p * x[(i + 1) % n] + q * x[(i + 2) % n]
        total += x[i] / denom if denom != 0.0 else np.inf
    return total


def _to_bool(value: object) -> bool | None:
    """Map the workbooks' 是/否 (Yes/No) flags onto booleans."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if text in TRUE_TOKENS:
        return True
    if text in FALSE_TOKENS:
        return False
    return None


def _detect_language(sheet_names: list[str]) -> str | None:
    for lang, names in SHEETS.items():
        if names["runs"] in sheet_names:
            return lang
    return None


def _read_case(path: Path):
    """Read one workbook: (header fields, run table, stored solution, language)."""
    with pd.ExcelFile(path) as xl:
        lang = _detect_language(list(xl.sheet_names))
        if lang is None:
            return {}, None, None, None
        key = SHEETS[lang]

        info: dict[str, object] = {}
        if key["summary"] in xl.sheet_names:
            raw = xl.parse(key["summary"])
            if raw.shape[1] >= 2:
                info = dict(zip(raw.iloc[:, 0].astype(str).str.strip(),
                                raw.iloc[:, 1]))
        runs = xl.parse(key["runs"]) if key["runs"] in xl.sheet_names else None
        sol = xl.parse(key["sol"]) if key["sol"] in xl.sheet_names else None
    return info, runs, sol, lang


def _solution_x(sol: pd.DataFrame | None, n: int) -> np.ndarray | None:
    """Pull the x_i column out of an 'Optimal Solution' sheet."""
    if sol is None or sol.empty:
        return None
    col = next((c for c in sol.columns if "y_i²" in str(c) or "yi²" in str(c)), None)
    if col is None:
        col = sol.columns[2] if sol.shape[1] > 2 else None
    if col is None:
        return None
    x = pd.to_numeric(sol[col], errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return x if x.size >= n else None


def collect(results_dir: Path):
    """Return (per-run frame, verified best-per-case frame, language counts)."""
    paths = sorted(results_dir.glob("shapiro_n*_p*.xlsx"))
    if not paths:
        raise SystemExit(f"no result workbooks found under {results_dir}")

    per_run: list[dict] = []
    per_case: list[dict] = []
    seen: Counter = Counter()
    skipped: list[str] = []

    for path in paths:
        info, runs, sol, lang = _read_case(path)
        if lang is None:
            skipped.append(path.name)
            continue
        seen[lang] += 1
        cols = COLS[lang]

        n, p = info.get(cols["n"]), info.get(cols["p"])
        m = FILENAME_RE.search(path.name)
        if n is None and m:
            n = int(m.group("n"))
        if p is None and m:
            p = float(m.group("p"))
        try:
            n, p = int(float(n)), float(p)
        except (TypeError, ValueError):
            skipped.append(path.name)
            continue

        # --- independent verification of the stored point -------------------
        reported = info.get(cols["best"])
        try:
            reported = float(reported)
        except (TypeError, ValueError):
            reported = None
        x = _solution_x(sol, n)
        recomputed = objective(x, p) if x is not None else None
        consistent = (
            reported is not None and recomputed is not None
            and abs(recomputed - reported) <= TOL
        )

        per_case.append({
            "n": n, "p": p, "q": info.get(cols["q"]),
            "n_runs_declared": info.get(cols["runs"]),
            "total_iterations": info.get(cols["total_iter"]),
            "total_time_s": info.get(cols["total_time"]),
            "best_reported": reported,
            "best_recomputed": recomputed,
            "consistent": consistent,
            "file": path.name,
        })

        if runs is None or runs.empty:
            continue
        for _, row in runs.iterrows():
            per_run.append({
                "n": n, "p": p, "file": path.name,
                "run": row.get(cols["run"]),
                "iterations": row.get(cols["iter"]),
                "time_s": row.get(cols["time"]),
                "objective": row.get(cols["obj"]),
                "grad_norm": row.get(cols["grad"]),
                "converged_flag": _to_bool(row.get(cols["conv"])),
                "alternating_detected": _to_bool(row.get(cols["osc"])),
                "alternating_a": row.get(cols["osc_a"]),
                "alternating_b": row.get(cols["osc_b"]),
            })

    if skipped:
        print(f"  skipped {len(skipped)} unreadable workbook(s)", file=sys.stderr)

    runs_df = pd.DataFrame(per_run)
    cases_df = pd.DataFrame(per_case)
    if cases_df.empty:
        return runs_df, cases_df, seen

    # --- verified per-case aggregate -----------------------------------------
    # A case is summarised by the best objective *actually attained*, i.e. the
    # smallest value recomputed from a stored solution.  Workbooks that fail
    # verification contribute their recomputed value only, never their claim.
    cases = cases_df.copy()
    cases["verified_value"] = cases["best_recomputed"].where(
        cases["best_recomputed"].notna(), cases["best_reported"])

    grp = cases.groupby(["n", "p"])
    best_df = grp.agg(
        workbooks=("file", "nunique"),
        n_inconsistent=("consistent", lambda s: int((~s.astype(bool)).sum())),
        runs_total=("n_runs_declared", "sum"),
        objective_min=("verified_value", "min"),
        objective_min_reported=("best_reported", "min"),
        total_time_s=("total_time_s", "sum"),
    ).reset_index()

    if not runs_df.empty:
        extra = runs_df.groupby(["n", "p"]).agg(
            runs_from_tables=("objective", "size"),
            grad_norm_min=("grad_norm", "min"),
            iterations_median=("iterations", "median"),
        ).reset_index()
        best_df = best_df.merge(extra, on=["n", "p"], how="left")

    best_df["gap_to_n"] = best_df["objective_min"] - best_df["n"]
    best_df["beats_n"] = best_df["gap_to_n"] < -1e-6
    best_df = best_df.sort_values(["n", "p"]).reset_index(drop=True)
    return runs_df, best_df, seen


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results_dir", type=Path,
                    help="directory holding the timestamped result workbooks")
    ap.add_argument("-o", "--out", type=Path, default=Path("results/summary"),
                    help="output directory for the CSV summaries")
    args = ap.parse_args(argv)

    print(f"scanning {args.results_dir} ...")
    runs_df, best_df, seen = collect(args.results_dir)

    args.out.mkdir(parents=True, exist_ok=True)
    runs_path, best_path = args.out / "runs_all.csv", args.out / "best_by_case.csv"
    runs_df.to_csv(runs_path, index=False, encoding="utf-8-sig")
    best_df.to_csv(best_path, index=False, encoding="utf-8-sig")

    langs = ", ".join(f"{k}:{v}" for k, v in sorted(seen.items())) or "none"
    print(f"  workbooks by language: {langs}")
    print(f"  {len(runs_df)} runs over {len(best_df)} (n, p) cases")
    if not best_df.empty:
        print(f"  inconsistent workbooks: {int(best_df['n_inconsistent'].sum())}")
        print(f"  cases where the verified minimum is below n: "
              f"{int(best_df['beats_n'].sum())}")
    print(f"  wrote {runs_path}")
    print(f"  wrote {best_path}")
    print(f"  generated {datetime.now():%Y-%m-%d %H:%M}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
