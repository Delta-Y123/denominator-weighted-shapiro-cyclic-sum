#!/usr/bin/env python3
"""Non-interactive command-line front end for the RBFGS solver.

``RBFGS_Algorithm.py`` (and its Chinese counterpart ``WeightedShapiro.py``) are
driven by an interactive ``input()`` menu, which makes them awkward to use from
a script, a notebook, or a CI job.  This wrapper calls the same functions with
arguments taken from the command line, so a case can be reproduced with a single
command.

It does not modify or duplicate the algorithm: it imports
``test_weighted_shapiro_with_restarts`` from ``RBFGS_Algorithm.py``.

Examples
--------

Reproduce the worked example of the paper, P(6, 0.3) = 30/7::

    python run_shapiro.py --n 6 --p 0.3

Sweep the weight for one dimension::

    python run_shapiro.py --n 25 --p 0.4 0.45 0.5 0.55 0.6

Verify the classical Shapiro counterexample at n = 14::

    python run_shapiro.py --n 14 --p 0.5 --restarts 10

Results are written to ``<outdir>/shapiro_results/`` as timestamped ``.xlsx``
workbooks, exactly as the interactive menu does.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Select a non-interactive backend *before* the solver imports pyplot, so that
# the plt.show() calls inside it return immediately instead of blocking.
import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (after the backend switch)

# --- forward-compatibility shim -------------------------------------------
# RBFGS_Algorithm.py does `from matplotlib.cm import get_cmap`.  That alias was
# deprecated in matplotlib 3.7 and removed in 3.11, so the solver cannot even be
# imported on a current matplotlib.  Re-instating the alias keeps the original
# file untouched and lets this wrapper run on both old and new releases.
import matplotlib.cm  # noqa: E402

if not hasattr(matplotlib.cm, "get_cmap"):
    matplotlib.cm.get_cmap = matplotlib.colormaps.get_cmap

# Make the module importable no matter where the script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import RBFGS_Algorithm as solver  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run the Riemannian BFGS solver for the denominator-weighted "
                    "Shapiro cyclic sum without the interactive menu.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples")[-1],
    )
    ap.add_argument("--n", type=int, nargs="+", required=True,
                    metavar="N", help="dimension(s), n >= 3")
    ap.add_argument("--p", type=float, nargs="+", required=True,
                    metavar="P", help="weight(s) in (0, 1); q = 1 - p")
    ap.add_argument("--restarts", type=int, default=10,
                    help="maximum number of perturbation restarts for even n "
                         "(default: 10)")
    ap.add_argument("--outdir", type=Path, default=Path("."),
                    help="working directory; workbooks go to "
                         "<outdir>/shapiro_results (default: current directory)")
    ap.add_argument("--no-excel", action="store_true",
                    help="skip writing the .xlsx workbooks")
    ap.add_argument("--show", action="store_true",
                    help="keep the interactive matplotlib backend and display "
                         "the convergence figures")
    ap.add_argument("--module", default="RBFGS_Algorithm",
                    choices=["RBFGS_Algorithm", "WeightedShapiro"],
                    help="which solver module to drive (default: the English "
                         "RBFGS_Algorithm; WeightedShapiro is the Chinese variant)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # --module lets the Chinese variant be used without a second wrapper.
    global solver
    if args.module != "RBFGS_Algorithm":
        import importlib

        solver = importlib.import_module(args.module)

    bad_n = [n for n in args.n if n < 3]
    if bad_n:
        raise SystemExit(f"n must be >= 3 (got {bad_n})")
    bad_p = [p for p in args.p if not 0.0 < p < 1.0]
    if bad_p:
        raise SystemExit(f"p must lie strictly between 0 and 1 (got {bad_p})")

    args.outdir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.outdir)

    cases = [(n, p) for n in args.n for p in args.p]
    print(f"running {len(cases)} case(s); results -> "
          f"{Path.cwd() / 'shapiro_results'}\n")

    rows: list[dict] = []
    for i, (n, p) in enumerate(cases, start=1):
        print(f"### case {i}/{len(cases)}:  n={n}, p={p}\n")
        result = solver.test_weighted_shapiro_with_restarts(
            n=n, p=p, max_restarts=args.restarts,
            save_to_excel=not args.no_excel,
        )
        best = float(result["best_obj"])
        rows.append({
            "n": n,
            "p": p,
            "best_objective": best,
            "n_bound": n,
            "difference": best - n,
            "runs": result.get("num_runs"),
            "iterations": result.get("total_iterations"),
            "time_s": result.get("total_time"),
            "inequality_holds": result.get("inequality_satisfied"),
        })
        if args.show:
            plt.show()
        print()

    # --- summary table -------------------------------------------------------
    width = max(len("best objective"), 14)
    print("=" * 78)
    print(f"{'n':>5}  {'p':>8}  {'best objective':>{width}}  {'n':>6}  "
          f"{'difference':>12}  {'runs':>5}  {'time (s)':>9}")
    print("-" * 78)
    for r in rows:
        print(f"{r['n']:>5}  {r['p']:>8.4f}  {r['best_objective']:>{width}.8f}  "
              f"{r['n_bound']:>6}  {r['difference']:>12.3e}  "
              f"{str(r['runs']):>5}  {float(r['time_s']):>9.3f}")
    print("=" * 78)

    below = [r for r in rows if r["difference"] < -1e-6]
    if below:
        print(f"{len(below)} case(s) reached a value strictly below n, i.e. a "
              f"counterexample to F(x, p) >= n:")
        for r in below:
            print(f"  n={r['n']}, p={r['p']:.4f}: {r['best_objective']:.8f} < {r['n']}")
    else:
        print("every case stayed at or above n (no counterexample found).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
