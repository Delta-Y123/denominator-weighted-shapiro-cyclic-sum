# 分母加权 Shapiro 循环和的研究与高效求解算法

**王安蕲** — 同济大学数学科学学院，上海 200092

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19228120.svg)](https://doi.org/10.5281/zenodo.19228120)
[![License: MIT](https://img.shields.io/badge/code%20license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](code/requirements.txt)

📄 [英文论文 (PDF)](paper/Denominator-Weighted-Shapiro-Cyclic-Sum.pdf) ·
🇨🇳 [中文版论文](paper/Denominator-Weighted-Shapiro-Cyclic-Sum-CN.pdf) ·
📦 [Zenodo 归档代码](https://doi.org/10.5281/zenodo.19228120) ·
🇬🇧 [English README](README.md)

---

## 内容简介

本仓库包含论文《Research on the Denominator-Weighted Shapiro Cyclic Sum and
Efficient Solution Algorithms》的全文、LaTeX 源码，以及配套的数值程序。

研究对象是**分母加权 Shapiro 循环和**

$$F(x,p)\;=\;\sum_{i=1}^{n}\frac{x_i}{p\,x_{i+1}+q\,x_{i+2}},
\qquad p+q=1,\quad 0\le p\le 1,\quad x_{n+1}=x_1,\ x_{n+2}=x_2,$$

以及不等式 $F(x,p)\ge n$。取 $p=q=\tfrac12$ 即回到经典的 **Shapiro 不等式**：该
问题历经三十余年，最终由 Troesch 完全解决 —— 当奇数 $3\le n\le 23$ 或偶数
$4\le n\le 12$ 时不等式成立；当偶数 $n\ge 14$ 或奇数 $n\ge 25$ 时存在反例。

本文有三方面贡献：

1. **极小点的结构刻画。** 把 Bushell–Ando 关于极小点的结构定理推广到分母加权
   情形：极小点可分解为若干"段"，各段的头尾比满足一条循环相等链；等长段之间
   成比例；具有相同零位模式的两个极小点必成比例。
2. **有限代数方法。** 把求 $P(n,p)=\min_x F(x,p)$ 归结为求解**有限个二元多项式
   方程组**——每个可达零位模式对应一个方程组，并给出其可行域 $UV_p$ 的显式描述。
   由此可手算出例如 $P(6,0.3)=30/7$。
3. **高效 RBFGS 算法。** 针对高维情形，通过代换 $x_i=y_i^2$ 把约束优化问题转化为
   单位球面 $\mathbb{S}^{n-1}$ 上的无约束问题，用带振荡解检测与扰动重启机制的
   黎曼 BFGS 方法求解。

理论上的核心结论是：**只要权重足够大，不等式对一切 $n\ge 3$ 成立。**

> **定理 3.18.** 对任意整数 $n\ge 3$，若 $\dfrac{\sqrt5-1}{2}\le p\le 1$，
> 则 $P(n,p)=\min_{x\in K_n}F(x,p)=n$。

## 主要结果一览

| 结论 | 内容（简述） |
| --- | --- |
| 定理 2.1 | 经典 Shapiro 不等式（$p=\tfrac12$）的 Troesch 三情形分类。 |
| 定理 2.2（Nguyen–Le） | $n=3$ 时恒成立；$n=4$ 时成立当且仅当 $\tfrac12\le p\le 1$；$n=5$ 的充分条件为 $\frac{5-\sqrt5}{10}\le p\le\frac{5+\sqrt5}{10}$；偶数 $n\ge 4$ 且 $0\le p<\tfrac12$ 时不成立。 |
| 定理 3.1 | 任意 $n\ge 3$、$0<p<1$，最小值 $P(n,p)$ 可达。 |
| 定理 3.2 | 递推估计：$p>\tfrac12$ 时 $P(n+2,p)\le P(n,p)+\frac{1}{1-p}$（$p\le\tfrac12$ 时为 $+2$）；$n$ 为奇数时 $P(n+1,p)\le P(n,p)+1$。 |
| 定理 3.3 | $P(n,p)$ 在 $p\in(0,1)$ 上连续。 |
| 定理 3.6、3.10 | 极小点满足尾/头比的循环相等链，在段的置换下不变，且满足 $T^2(x^\*)=x^\*$。 |
| 定理 3.12 | 同零位模式的两个极小点成比例；内部极小点必使 $F=n$，且只能是常值，或 $n$ 偶且 $p=\tfrac12$ 时的振荡型。 |
| 定理 3.14、3.17 | 每一段满足一个显式的特征型方程；边界极小点满足 $U(x^\*)\ge q$。 |
| **定理 3.18** | $\frac{\sqrt5-1}{2}\le p\le 1\ \Longrightarrow\ P(n,p)=n$，对一切 $n\ge 3$。 |
| §3.3.1 | 由有限个多项式方程组求 $P(n,p)$ 的三步流程。 |
| §4.2 | 带振荡检测与重启的黎曼 BFGS 算法流程（依据 Ring–Wirth、Huang–Gallivan–Absil）。 |

## 数值验证

`results/summary/` 中的汇总数据已与论文定理逐条核对。每一个数值都是**经过校验的**：
从工作簿中读回保存的极小点，用独立实现重新计算目标函数值（见
[`tools/verify_objective.py`](tools/verify_objective.py)）。

**定理 3.18** —— 归档数据中所有 35 个满足 $p\ge\frac{\sqrt5-1}{2}$ 的情形都给出
$P(n,p)=n$，最大偏差仅 $2.3\times10^{-12}$。

**经典 Shapiro 不等式（$p=\tfrac12$）** —— 归档的 45 个情形中有 44 个与 Troesch
的分类一致。唯一例外是 $n=101$：程序返回 $103.65>101$，即没有收敛到反例。这是
求解器的局限而非数学结论 —— 奇数 $n$ 只做一次随机起点、不重启，因此失手是可以
预期的。

部分数值（经校验，取自归档运行）：

| $n$ | $p$ | 校验后最小值 | $n$ | 说明 |
| ---: | ---: | ---: | ---: | --- |
| 4 | 0.3 | 2.8571428571 | 4 | 小于 $n$，与定理 2.2 相符 |
| 4 | 0.5 | 4.0000000000 | 4 | 取等 |
| 6 | 0.3 | 4.2857142857 | 6 | $=30/7$，即论文例 3.21 |
| 12 | 0.5 | 12.0000000000 | 12 | 不等式成立 |
| 14 | 0.5 | 13.9993053941 | 14 | 不等式失效，符合预期 |
| 23 | 0.5 | 23.0233788761 | 23 | 成立；程序未精确取到 $23$ |
| 25 | 0.5 | 24.9975582112 | 25 | 不等式失效，符合预期 |
| 100 | 0.5 | 99.4264914630 | 100 | 失效 |
| 250 | 0.5 | 249.1647197445 | 250 | 失效 |

**请注意。** 以上都是**数值最小化**结果，不是证明。略小于 $n$ 的数值对应的是真实
可行的点（独立重算已确认），且在 $p=\tfrac12$ 时与已知理论一致。但本仓库中的任何
内容都不应被理解成对某个不等式的证明。

### 数据完整性说明

`tools/verify_objective.py` 发现 246 个工作簿中有 **4 个**记录的最优目标值与该文件
内保存的解所实际对应的目标值不符 —— 例如
`shapiro_n14_p0.5000_20251223_160132.xlsx` 声称 $9.009$，但其保存的点实际算得
$1748.84$。这类文件在
[`results/summary/best_by_case.csv`](results/summary/best_by_case.csv) 的
`n_inconsistent` 列中被计数，并已从校验后的最小值中剔除。

## 目录结构

```
.
├── paper/
│   ├── Denominator-Weighted-Shapiro-Cyclic-Sum.pdf      英文论文
│   ├── Denominator-Weighted-Shapiro-Cyclic-Sum-CN.pdf   中文论文
│   └── source/                                          论文 LaTeX 源码
│       ├── main.tex、main.bbl、cas-refs.bib
│       └── figure/                                      main.tex 引用的图
├── code/
│   ├── python/
│   │   ├── RBFGS_Algorithm.py      求解器 + 交互菜单（英文）
│   │   ├── WeightedShapiro.py      求解器 + 交互菜单（中文）
│   │   ├── run_shapiro.py          非交互命令行入口  <-- 建议从这里开始
│   │   ├── Compute_f_i.py          用 sympy 符号计算 f_i
│   │   └── Plot_UV_Region.py       绘制可行域 UV_p
│   ├── matlab/
│   │   ├── generalN.m、matrix6.m   符号主子式计算
│   │   └── u_v_area.m、plot_region.m   绘制 (u, v) 区域
│   └── requirements.txt
├── figures/                        论文中的插图
├── results/
│   └── summary/
│       ├── best_by_case.csv        每个 (n, p) 一行，已校验
│       └── runs_all.csv            每次运行一行
├── tools/
│   ├── summarize_results.py        从原始工作簿生成上述 CSV
│   └── verify_objective.py         在保存的解上重算目标函数
├── CITATION.cff
├── LICENSE                         代码：MIT
└── LICENSE-PAPER.md                论文与插图的授权说明
```

## 快速开始

### 安装

```bash
git clone <本仓库地址>
cd denominator-weighted-shapiro-cyclic-sum
python -m pip install -r code/requirements.txt
```

Windows 上请先设置控制台编码。求解器会打印 `✓`、`✗`、`²` 等字符，它们无法用
GBK 代码页表示，重定向输出时会直接崩溃：

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:MPLBACKEND = "Agg"      # 无显示器 / 服务器环境
```

### 运行求解器（推荐：命令行入口）

`run_shapiro.py` 调用与交互菜单完全相同的函数，但参数从命令行取得：

```bash
# 复现论文中的算例 P(6, 0.3) = 30/7
python code/python/run_shapiro.py --n 6 --p 0.3

# 对某一维数扫描权重
python code/python/run_shapiro.py --n 25 --p 0.4 0.45 0.5 0.55 0.6

# 复核 n = 14 的经典反例
python code/python/run_shapiro.py --n 14 --p 0.5 --restarts 10

# 使用中文版求解器
python code/python/run_shapiro.py --module WeightedShapiro --n 6 --p 0.3
```

结果以带时间戳的 `.xlsx` 写入 `<outdir>/shapiro_results/`。

### 运行原始脚本

两个求解器脚本都由交互菜单驱动（`1` 单次测试、`2` 批量测试、`3` 退出）：

```bash
python code/python/RBFGS_Algorithm.py
# 依次输入： 1 ⏎ 100 ⏎ 0.5 ⏎ ⏎ 3 ⏎
```

也可以用管道非交互地喂入输入：

```bash
printf '1\n6\n0.3\n\n3\n' | python code/python/WeightedShapiro.py
printf '2\n3,0.5;4,0.5;5,0.5;6,0.5\n10\n3\n' | python code/python/WeightedShapiro.py
```

作为库调用（无菜单，也不会被 `plt.show()` 阻塞）：

```python
from WeightedShapiro import SphereManifold, MultiStartWeightedShapiroOptimizer

manifold = SphereManifold(50)
result = MultiStartWeightedShapiroOptimizer(manifold, 0.5).optimize_with_restarts(
    max_restarts=5
)
print(result["best_obj"])
```

### 符号计算与绘图辅助脚本

```bash
python code/python/Compute_f_i.py        # 输入 i (>= -1)，再选择是否代入 p 的数值
python code/python/Plot_UV_Region.py     # 输入 p 的值，例如 0.3
```

### MATLAB 部分

`code/matlab/*.m` 需要安装了 **Symbolic Math Toolbox** 的 MATLAB（用到 `syms`、
`simplify`、`nchoosek`、`isAlways`），GNU Octave 无法运行。`u_v_area.m` 与
`plot_region.m` 两个文件内容完全相同，且末尾有一句多余的顶层 `input()`；把它们
注释掉即可作为函数调用：

```matlab
p = 0.3; plot_region(p);
```

`generalN.m` 会枚举全部 $k$ 阶主子式并逐个化简行列式，因此大致只在
$n\lesssim 15$ 时可用。

### 重新生成汇总数据

```bash
python tools/summarize_results.py shapiro_results -o results/summary
python tools/verify_objective.py  shapiro_results --n 14 --p 0.5
```

## 输出格式

每次运行写一个含 5 个工作表的 Excel 文件；`RBFGS_Algorithm.py` 用英文表名，
`WeightedShapiro.py` 用中文表名。

| 英文表名 | 中文表名 | 每行代表 |
| --- | --- | --- |
| `Summary` | `汇总信息` | —（本次运行的键值概览） |
| `Run Details` | `各次运行详情` | 一次优化运行 / 一次重启 |
| `Optimal Solution` | `最优解详细信息` | 一个坐标 $i=1,\dots,n$ |
| `Statistics` | `统计信息` | —（键值统计量） |
| `Convergence History` | `收敛历史` | 一次迭代（仅当 $n\le 300$ 时写出，每次运行只保留前 200 个点） |

需要注意：表中多数数值是以格式化**文本**而非数字写入的。原始结果文件不纳入 git，
只有 `results/summary/` 下派生的 CSV 会提交。

## 算法要点

- **参数化。** $x_i=y_i^2$ 把非负象限映到单位球面 $\mathbb{S}^{n-1}$，约束变为
  $\lVert y\rVert=1$。
- **黎曼梯度。** 先求 $F(y^2,p)$ 的欧氏梯度，再投影到切空间，
  $\operatorname{grad}f=P_y(\nabla f)=v-\langle y,v\rangle y$。
- **收缩与传输。** $R_y(\eta)=(y+\eta)/\lVert y+\eta\rVert$，以及球面上的平行移动。
- **RBFGS。** Armijo 回溯线搜索 + 逆 Hessian 的 BFGS 更新；曲率条件不满足时把
  Hessian 重置为单位阵。
- **振荡处理**（仅偶数 $n$）。若某次运行停在二周期模式 $(a,b,a,b,\dots)$ 且目标值
  仍不小于 $n$，则对该点施加 5% 高斯扰动后重启，最多 10 次。奇数 $n$ 只跑一次
  随机起点。
- **数值保护。** 分母下限 $10^{-10}$、梯度裁剪到 $\pm10^3$、安全范数。

目标函数与梯度的每次迭代代价为 $O(n)$，BFGS 的矩阵向量积为 $O(n^2)$。因此
$n\approx 250$ 以内都很轻松，超过 $n\sim 10^3$ 后 $n\times n$ 稠密 Hessian 近似
成为瓶颈。

## 已知局限

以下问题在此明确记录，以免结果被过度解读：

- 驱动脚本默认是交互式的；`run_shapiro.py` 正是为了绕开这一点而存在。
- 求解器脚本 import 了 `matplotlib.cm.get_cmap`，而该别名在 matplotlib 3.11 中
  已被移除。
- 脚本从未调用 `savefig`；`figures/` 中的 PNG 是手工导出的。
- BFGS 更新中的平行移动把**当前**步长与**上一次**方向混用，因此割线条件并不严格
  成立。
- 梯度裁剪与"曲率不足即把 $H$ 重置为 $I$"都是未加说明的启发式处理。
- `RBFGS_Algorithm.py` 以 $\lVert g\rVert<10^{-4}$ 判定收敛，
  `WeightedShapiro.py` 用的是 $10^{-6}$，因此同一次运行两版给出的 `Converged`
  标志可能不同。
- 有 4 个归档工作簿未通过上述内部一致性检查。

## 引用

若使用本工作，请同时引用论文与归档代码，详见 [`CITATION.cff`](CITATION.cff)。

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

## 致谢

感谢本项目指导教师、数学分析课程教师潘生亮教授的悉心指导，感谢同济大学数学科学
学院提供的良好学术氛围。本工作受同济大学大学生创新训练计划（SITP）资助，项目
编号 X2025634。

## 授权

- **代码**（`code/`）：[MIT](LICENSE)，与 Zenodo 归档版本一致。
- **论文、LaTeX 源码与插图**（`paper/`、`figures/`）：见
  [LICENSE-PAPER.md](LICENSE-PAPER.md)，版权归作者所有。
