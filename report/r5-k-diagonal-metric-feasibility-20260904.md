# R5-K：保持三次项耗散的对角度量没有找到可行解

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-09-04
- Verification Status: VERIFIED — NUMERICAL FEASIBILITY SCREEN FAILED
- Version Label: exp_result_v1
- Formal Code Commit: `eea63f2b62e404e08210f5f5a1b554ebc7c8a90c`
- Result SHA-256: `1bd1a9974315949783a48655d942fc9796a88e9b478b9c979132648649a8ebbf`
- Node: RTX 2060 Mobile Linux

## 结论

本实验严格沿用“稳定线性骨架 + 真实变换后非线性余项”的分解，但把坐标限制为

\[
T_0=M_h^{1/2},\qquad M_h=\operatorname{diag}(m_1,\ldots,m_n)>0.
\]

这种坐标只给各空间点加权而不混合位置，因此 Allen--Cahn 的真实三次增量满足

\[
e^\top M_h\big((u+e)^3-u^3\big)
=\sum_i m_i e_i^2(3u_i^2+3u_ie_i+e_i^2)\geq0
\]

对所有状态和误差都成立。只要同时找到三传感器增益 (B_h)，使线性误差系统在该度量中
收缩，就能立即得到全局半离散非线性收缩结论。

在冻结的 (\nu=0.005)、三传感器、(n=31) 问题上，半正定可行性筛选从

\[
\kappa(M_h)\leq4
\]

逐级放宽到

\[
\kappa(M_h)\leq10^6,
\]

仍没有任何候选通过恢复到原物理尺度后的特征值复核。因此，这个最简单的“逐点加权”
坐标不能作为下一步证明工具。

这里的严格表述是“冻结的数值筛选未找到可行解”，不是仅凭 SCS 状态宣称一个解析
不可行定理。后续 R5-L 将用不依赖优化器的传感器间隙构造，判断这种失败是否来自三传感器
几何本身，并确定固定线性 (T_0) 能够证明的真实范围。

## 数值结果

请求线性收缩率为

\[
\alpha=0.1\nu\pi^2=0.004934802200544679.
\]

| 条件数上界 | 求解器状态 | 原尺度复核 | 复核线性收缩率 | (\ker C_h) 必要条件最大值 |
|---:|---|---|---:|---:|
| 4 | infeasible | 无候选 | — | — |
| 16 | infeasible | 无候选 | — | — |
| 64 | infeasible | 无候选 | — | — |
| 256 | infeasible | 无候选 | — | — |
| 1024 | optimal_inaccurate | 失败 | -0.386472 | +0.760448 |
| 4096 | optimal_inaccurate | 失败 | -0.432674 | +0.837203 |
| 16384 | optimal_inaccurate | 失败 | -0.492546 | +0.837069 |
| 65536 | optimal_inaccurate | 失败 | -0.506414 | +0.837071 |
| (10^6) | optimal_inaccurate | 失败 | -0.456010 | +0.836833 |

后五个求解达到迭代上限后只返回不精确候选。它们的线性收缩率均为负，且在观测零空间上
仍存在正增长方向，所以没有把求解器的 `optimal_inaccurate` 标签误记为成功。随机三次项
回归检查全部保持正号，这说明失败确实发生在线性可观测几何，而不是三次项恒等式或实现。

## 对所提路线的回答

R5-K 没有要求

\[
T_0(e^3)=(T_0e)^3.
\]

它保留了变换后的真实非线性，并试图用其真实能量符号完成证明。因此路线本身没有被否定；
被否定的是一个过于简单的 (T_0)：只做逐点正缩放、不允许空间混合。

R5-J 与 R5-K 合起来说明了两端的矛盾：

1. Sylvester 坐标混合过强，线性部分漂亮，但真实非线性会在该能量中反耗散；
2. 正对角坐标完整保留非线性耗散，但没有足够几何自由度压稳三传感器线性系统。

下一步不能盲目在两者之间调参数，而应先检查是否存在由传感器空隙导致的固定线性坐标
全局障碍；若存在，则正确目标应改为对指定状态类的条件化证明，或者使用状态依赖坐标，
而不是继续追求三传感器下对全部状态的固定 (T_0) 全局证书。

## 完整性与复现

- 第一次 CLARABEL 调用在首个半正定问题上抛出 `SolverError`，被记录为程序管线失败，
  没有生成科学结论；
- 修订后 SCS 正式运行退出码为 0；
- 每个可返回矩阵的候选都在原尺度独立重算 Lyapunov 特征值、收缩率、闭环谱和
  (\ker C_h) 必要条件；
- 不使用训练数据、validation、locked test 或 checkpoint；
- 正式结果保留在远程：
  `/home/wjj/work/wt/phd-project1-codex/tmp/r5-route-worktrees/backstepping-nonlinear-remainder/out/2026-09-04-r5-k-diagonal-n31-scs-eea63f2/result.json`；
- 远程运行 commit 与正式代码 commit 一致。
