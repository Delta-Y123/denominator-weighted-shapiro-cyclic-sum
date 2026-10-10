# Research on the Denominator-Weighted Shapiro Cyclic Sum and Efficient Solution Algorithms

**Anqi Wang** — School of Mathematical Sciences, Tongji University, Shanghai 200092, China

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19228120.svg)](https://doi.org/10.5281/zenodo.19228120)
[![License: MIT](https://img.shields.io/badge/code%20license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](code/requirements.txt)

📄 [Paper (English, PDF)](paper/Denominator-Weighted-Shapiro-Cyclic-Sum.pdf) ·
🇨🇳 [中文版论文](paper/Denominator-Weighted-Shapiro-Cyclic-Sum-CN.pdf) ·
📦 [Archived code on Zenodo](https://doi.org/10.5281/zenodo.19228120) ·
🇨🇳 [中文说明](README.zh-CN.md)

---

## Summary

This repository contains the manuscript and the numerical code for a study of the
**denominator-weighted Shapiro cyclic sum**

$$F(x,p)\;=\;\sum_{i=1}^{n}\frac{x_i}{p\,x_{i+1}+q\,x_{i+2}},
\qquad p+q=1,\quad 0\le p\le 1,\quad x_{n+1}=x_1,\ x_{n+2}=x_2,$$

and of the inequality $F(x,p)\ge n$. Setting $p=q=\tfrac12$ recovers the classical
**Shapiro inequality**, which resisted proof for three decades and was finally
settled by Troesch: it holds for odd $3\le n\le 23$ and even $4\le n\le 12$, and
fails for even $n\ge 14$ and odd $n\ge 25$.

The work makes three contributions:

1. **Structure of minimizers.** The Bushell–Ando structural characterization of
   minimizers is extended to the denominator-weighted case: minimizers are shown
   to organise into *segments* whose heads and tails satisfy a chain of equal
   ratios, segments of equal length are proportional to one another, and two
   minimizers with the same zero pattern must be proportional.
2. **A finite algebraic method.** Computing $P(n,p)=\min_x F(x,p)$ is reduced to
   solving **finitely many polynomial equation systems** in two variables, one
   system per admissible zero pattern, each confined to an explicitly described
   feasible region $UV_p$. This yields, for example, $P(6,0.3)=30/7$ by hand.
3. **An efficient RBFGS algorithm.** For high dimensions, the substitution
   $x_i=y_i^2$ turns the constrained problem into an unconstrained one on the unit
   sphere $\mathbb{S}^{n-1}$, solved by a Riemannian BFGS method with
   oscillatory-solution detection and perturbation restarts.

The headline theoretical result is that the inequality holds **for every** $n\ge 3$
once the weight is large enough:

> **Theorem 3.18.** For every integer $n\ge 3$, if $\dfrac{\sqrt5-1}{2}\le p\le 1$,
> then $P(n,p)=\min_{x\in K_n}F(x,p)=n$.

## Highlights (as submitted)

- Establishes key properties and characteristic results of the denominator-weighted
  Shapiro cyclic sum.
- Extends the Bushell–Ando structural characterization of minimizers to the
  denominator-weighted case.
- Reduces the minimization problem to solving finitely many polynomial equation
  systems.
- An efficient RBFGS algorithm on the unit sphere is proposed to minimize the
  Shapiro cyclic sum.

## Main results

| Result | Statement (abridged) |
| --- | --- |
| Thm 2.1 | Troesch's three-case classification of the classical Shapiro inequality ($p=\tfrac12$). |
| Thm 2.2 (Nguyen–Le) | For $n=3$ the inequality always holds; for $n=4$ it holds iff $\tfrac12\le p\le 1$; for $n=5$ a sufficient condition is $\frac{5-\sqrt5}{10}\le p\le\frac{5+\sqrt5}{10}$; for even $n\ge 4$ and $0\le p<\tfrac12$ it fails. |
| Thm 3.1 | For every $n\ge 3$ and $0<p<1$ the minimum $P(n,p)$ is attained. |
| Thm 3.2 | Recursive bounds: $P(n+2,p)\le P(n,p)+\frac{1}{1-p}$ for $p>\tfrac12$ (and $+2$ for $p\le\tfrac12$); for odd $n$, $P(n+1,p)\le P(n,p)+1$. |
| Thm 3.3 | $P(n,p)$ is continuous in $p$ on $(0,1)$. |
| Thm 3.6, 3.10 | Minimizers satisfy a cyclic chain of equal tail/head ratios, are invariant under permuting segments, and satisfy $T^2(x^\*)=x^\*$. |
| Thm 3.12 | Two minimizers sharing a zero pattern are proportional; an interior minimizer forces $F=n$ and must be constant, or oscillatory when $n$ is even and $p=\tfrac12$. |
| Thm 3.14, 3.17 | Each segment satisfies an explicit eigenvalue-type equation; a boundary minimizer satisfies $U(x^\*)\ge q$. |
| **Thm 3.18** | $\frac{\sqrt5-1}{2}\le p\le 1\ \Longrightarrow\ P(n,p)=n$ for all $n\ge 3$. |
| §3.3.1 | The three-step procedure that evaluates $P(n,p)$ from finitely many polynomial systems. |
| §4.2 | The Riemannian BFGS procedure with oscillatory detection and restart (following Ring–Wirth and Huang–Gallivan–Absil). |

## Numerical verification

The archive of runs in `results/summary/` was re-checked against the paper's
theorems. Every case is *verified*: the stored minimizer is read back from the
workbook and the objective is recomputed independently (see
[`tools/verify_objective.py`](tools/verify_objective.py)).

**Theorem 3.18** — all 35 cases in the archive with $p\ge\frac{\sqrt5-1}{2}$ return
$P(n,p)=n$.

## Repository layout

```
.
├── paper/
│   ├── Denominator-Weighted-Shapiro-Cyclic-Sum.pdf      manuscript (English)
│   ├── Denominator-Weighted-Shapiro-Cyclic-Sum-CN.pdf   manuscript (Chinese)
│   └── source/                                          LaTeX source of the manuscript
│       ├── main.tex, main.bbl, cas-refs.bib
│       └── figure/                                      figure files used by main.tex
├── code/
│   ├── python/
│   │   ├── RBFGS_Algorithm.py      solver + interactive menu (English)
│   │   ├── WeightedShapiro.py      solver + interactive menu (Chinese)
│   │   ├── run_shapiro.py          non-interactive CLI front end  <-- start here
│   │   ├── Compute_f_i.py          symbolic computation of the f_i (sympy)
│   │   └── Plot_UV_Region.py       draws the feasible region UV_p
│   ├── matlab/
│   │   ├── generalN.m, matrix6.m   symbolic principal minors
│   │   └── u_v_area.m, plot_region.m   plot the (u, v) region
│   └── requirements.txt
├── figures/                        figures as they appear in the manuscript
├── results/
│   └── summary/
│       ├── best_by_case.csv        one row per (n, p), verified
│       └── runs_all.csv            one row per individual run
├── tools/
│   ├── summarize_results.py        build the CSVs from the raw workbooks
│   └── verify_objective.py         recompute F at a stored solution
├── CITATION.cff
├── LICENSE                         MIT (code)
└── LICENSE-PAPER.md                terms for the manuscript and figures
```

## Getting started

### Install

```bash
git clone https://github.com/Delta-Y123/denominator-weighted-shapiro-cyclic-sum.git
cd denominator-weighted-shapiro-cyclic-sum
python -m pip install -r code/requirements.txt
```

On Windows, set the console encoding first — the solver prints `✓`, `✗` and `²`,
which are not representable in the legacy GBK code page and will crash a
redirected stdout:

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:MPLBACKEND = "Agg"      # headless / no display
```

### Run the solver (recommended: the CLI)

`run_shapiro.py` drives the same functions as the interactive menu, but takes its
arguments from the command line:

```bash
# reproduce the paper's worked example, P(6, 0.3) = 30/7
python code/python/run_shapiro.py --n 6 --p 0.3

# sweep the weight for one dimension
python code/python/run_shapiro.py --n 25 --p 0.4 0.45 0.5 0.55 0.6

# verify the classical counterexample at n = 14
python code/python/run_shapiro.py --n 14 --p 0.5 --restarts 10

# use the Chinese variant of the solver
python code/python/run_shapiro.py --module WeightedShapiro --n 6 --p 0.3
```

Results are written to `<outdir>/shapiro_results/` as timestamped `.xlsx`
workbooks.

### Run the original scripts

Both solver scripts are driven by an interactive menu (`1` single test,
`2` batch test, `3` exit):

```bash
python code/python/RBFGS_Algorithm.py
# then: 1 ⏎ 100 ⏎ 0.5 ⏎ ⏎ 3 ⏎
```

They can be fed non-interactively too:

```bash
printf '1\n6\n0.3\n\n3\n' | python code/python/WeightedShapiro.py
printf '2\n3,0.5;4,0.5;5,0.5;6,0.5\n10\n3\n' | python code/python/WeightedShapiro.py
```

As a library (no menu, no blocking `plt.show()`):

```python
from WeightedShapiro import SphereManifold, MultiStartWeightedShapiroOptimizer

manifold = SphereManifold(50)
result = MultiStartWeightedShapiroOptimizer(manifold, 0.5).optimize_with_restarts(
    max_restarts=5
)
print(result["best_obj"])
```

### Symbolic and plotting helpers

```bash
python code/python/Compute_f_i.py        # Enter i (>= -1), then y/n for substituting p
python code/python/Plot_UV_Region.py     # Enter the value of p, e.g. 0.3
```

### MATLAB

`code/matlab/*.m` require MATLAB with the **Symbolic Math Toolbox** (they use
`syms`, `simplify`, `nchoosek`, `isAlways`); GNU Octave will not run them.
`u_v_area.m` and `plot_region.m` are identical files that end with a stray
top-level `input()`; comment out those last lines to call them as functions:

```matlab
p = 0.3; plot_region(p);
```

`generalN.m` enumerates every $k$-th order principal minor and simplifies each
determinant, so it is only practical for roughly $n\lesssim 15$.

### Rebuild the result summaries

```bash
python tools/summarize_results.py shapiro_results -o results/summary
python tools/verify_objective.py  shapiro_results --n 14 --p 0.5
```

## Output format

Each run writes one workbook with five sheets; `RBFGS_Algorithm.py` names them in
English and `WeightedShapiro.py` in Chinese.

| English | Chinese | One row per |
| --- | --- | --- |
| `Summary` | `汇总信息` | — (key/value header of the run) |
| `Run Details` | `各次运行详情` | one optimisation run / restart |
| `Optimal Solution` | `最优解详细信息` | one coordinate $i=1,\dots,n$ |
| `Statistics` | `统计信息` | — (key/value aggregates) |
| `Convergence History` | `收敛历史` | one iteration (only written when $n\le 300$, first 200 points per run) |

Note that most numeric cells are written as formatted *text* rather than numbers.
Result files are not tracked in git; only the derived CSVs under `results/summary/`
are.

## Algorithm, briefly

- **Parameterisation.** $x_i=y_i^2$ maps the nonnegative orthant onto the unit
  sphere $\mathbb{S}^{n-1}$; the constraint becomes $\lVert y\rVert=1$.
- **Riemannian gradient.** The Euclidean gradient of $F(y^2,p)$ is projected onto
  the tangent space, $\operatorname{grad}f=P_y(\nabla f)=v-\langle y,v\rangle y$.
- **Retraction / transport.** $R_y(\eta)=(y+\eta)/\lVert y+\eta\rVert$ and the
  sphere's parallel transport.
- **RBFGS.** Armijo backtracking line search, inverse-Hessian BFGS update, with the
  Hessian reset to $I$ when the curvature condition fails.
- **Oscillation handling** (even $n$ only). Runs that settle on a two-periodic
  pattern $(a,b,a,b,\dots)$ are detected and, if the objective is still at least
  $n$, restarted from a 5 % Gaussian perturbation of that point, at most 10 times.
  Odd $n$ uses a single random start.
- **Numerical guards.** Denominator floor $10^{-10}$, gradient clipping to
  $\pm10^3$, and a safe norm.

Cost is $O(n)$ per iteration for the objective and gradient and $O(n^2)$ for the
BFGS matrix-vector product, so runs up to $n\approx 250$ are routine and the
$n\times n$ dense Hessian approximation becomes the bottleneck beyond
$n\sim 10^3$.

## Known limitations

These are documented so that results are not over-read:

- The drivers are interactive by default; `run_shapiro.py` exists to work around
  that.
- `matplotlib.cm.get_cmap` is imported by the solver scripts and was removed in
  matplotlib 3.11.
- The scripts never call `savefig`; the PNGs in `figures/` were exported by hand.
- The parallel transport in the BFGS update reuses the current step size with the
  previous direction, so the secant condition is not exactly satisfied.
- The gradient clipping and the unconditional reset of $H$ to $I$ are
  undocumented heuristics.
- `RBFGS_Algorithm.py` declares convergence at $\lVert g\rVert<10^{-4}$ while
  `WeightedShapiro.py` uses $10^{-6}$, so the two report different `Converged`
  flags for the same run.
- Four archived workbooks fail the internal consistency check described above.

## Citation

If you use this work, please cite both the manuscript and the archived code.
See [`CITATION.cff`](CITATION.cff).

```bibtex
@software{wang2026shapirocode,
  author    = {Wang, Anqi},
  title     = {Code for the Minimum of the Denominator-Weighted Shapiro Cyclic
               Sum via RBFGS},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0},
  doi       = {10.5281/zenodo.19228120}
}
```

## Acknowledgements

The author thanks Professor Shengliang Pan, supervisor of this research project
and instructor of the Mathematical Analysis course, for his careful guidance, and
the School of Mathematical Sciences, Tongji University, for its academic
atmosphere. This work was supported by the Tongji University Student Innovation
Training Program (SITP) under No. X2025634.

## License

- **Code** (`code/`): [MIT](LICENSE), matching the archived Zenodo release.
- **Manuscript, LaTeX source and figures** (`paper/`, `figures/`): see
  [LICENSE-PAPER.md](LICENSE-PAPER.md). All rights reserved by the author.
