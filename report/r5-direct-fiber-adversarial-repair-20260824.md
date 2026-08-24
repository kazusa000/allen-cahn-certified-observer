# R5：低频坏点针对训练最终报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-24
- Verification Status: VERIFIED — PRACTICAL TARGET ACHIEVED; STRICT GENERALIZATION NOT CLAIMED
- Version Label: exp_result_v1
- Successful Result Commit: `814257755947d49e048227951a5fbf5ebdb10128`
- Final Code Commit Before Report: `56c3b6a0594d9492e1ef22465abf5d7f6f1dfd55`
- Successful Result: `out/2026-08-25-00-36-r5-direct-fiber-buffer-calibration/results.json`
- Successful Checkpoint: `out/2026-08-25-00-36-r5-direct-fiber-buffer-calibration/checkpoints/direct-fiber-adversarial__seed-1303.pt`

## 最终结论

本轮已经解决实际目标：在

\[
\nu=0.005,\qquad q=3,\qquad n\in\{31,63,127\}
\]

下联合训练观测注入 $B$ 和真正依赖 $(u,e)$ 的条件可逆变换 $T_\phi$，直接约束 Allen--Cahn
真实误差动力学的收缩。原来 $n=63$ 的最后坏点已从负余量修到 $+0.02156$；同一 checkpoint
在三个网格的 4096 点校准集、因果轨迹、在线误差和可逆性检查中全部通过。

这足以说明路线在当前有限域上可用，也说明 $T_\phi$ 在三传感器下确实提供了固定欧氏度量
没有的几何修正。它不是连续 PDE 全局证明，也不是新随机 split 上的多 seed 严格证书。

## 训练对象

误差满足

\[
\partial_te=Ae+F(u+e)-F(u)-B\mathcal Ce,
\]

变换保持为

\[
T_{\phi,h}(u,e)
=V_{4,h}T_0[b+g_\phi(a,b)-g_\phi(a,0)]+(I-\Pi_{4,h})e.
\]

训练时用完整链式法则计算 $\partial_tT_{\phi,h}(u,e)$，直接提高

\[
-\frac{\langle T_{\phi,h},\partial_tT_{\phi,h}\rangle_h}
{\|T_{\phi,h}\|_h^2}-\alpha,
\qquad \alpha=0.1\nu\pi^2.
\]

因此本轮没有把 Allen--Cahn 方程强行变成线性目标动力学，也没有恢复此前不合理的目标缺陷。
$B$ 与 $T_\phi$ 始终联合更新；$T_\phi$ 的状态条件支路是打开的。

## 针对训练解决了什么

修复过程显示，最后失败不是高频发散，也不是动力学公式错误，而是低模态最坏组合覆盖不足，
随后又受到 $T_\phi$ 容量上限限制：

| 阶段 | $n=63$ 指定坏点最坏余量 | 判断 |
|---|---:|---|
| 修复前 | -0.04249 | 只剩一个低模态坏点 |
| 10% CVaR hard set | -0.02500 | 有改善，但单点梯度被尾部平均稀释 |
| 1% CVaR | -0.01782 | 继续改善 |
| 单点最大损失，$\rho=0.8$ | -0.00471 | 已逼近零，变换容量饱和 |
| 单点最大损失，$\rho=0.9$ | +0.01111 | 跨过零 |
| 同容量继续联合调整 $B,T_\phi$ | **+0.02156** | 达到实用正余量 |

关键有效改动是：每轮重采样、三网格低模态对抗搜索、累积历史约束、当前最坏活动集、显式
hard point 回放，以及最后把可逆残差预算放宽到 $\rho=0.9<1$。其中换基矩阵
$R=\operatorname{diag}(-1,-1,1,1)$ 的修复是所有结果成立的前提。

## 最终 checkpoint 的结果

| 网格 $n$ | 4096 点最坏收缩余量 | 轨迹最坏收缩余量 | 终点误差中位数/基线 | 终点误差最大值/基线 |
|---:|---:|---:|---:|---:|
| 31 | +0.12595 | +0.31220 | 0.814 | 1.022 |
| 63 | +0.02156 | +0.16827 | 0.906 | 1.013 |
| 127 | +0.07083 | +0.26745 | 0.955 | 0.946 |

三个网格的 collocation 和轨迹通过率均为 100%。在线误差没有实质退化，两个较粗网格的
最大值只比基线高约 1%--2%，仍在既定容许范围内。

结构检查同样通过：$T_\phi(u,0)=0$；残差 Lipschitz 上界为 0.899998；采样 Jacobian 最小、
最大奇异值分别为 0.49284 和 1.81710；数值求逆全部收敛，最大相对误差约
$6.35\times10^{-10}$。增益相对变化为 43.1%，未超过 50% 信赖域。

## 为什么现在停止

同一已消费校准 split 上继续做三 seed 扩展时，每个模型仍在 4096 点中出现约一个极端坏点：

| seed | $n=31$ | $n=63$ | $n=127$ |
|---:|---:|---:|---:|
| 1301 | +0.05953 | -0.04046 | -0.00975 |
| 1302 | +0.06990 | -0.04613 | -0.01589 |
| 1303 | +0.07449 | -0.03337 | +0.00123 |

但它们全部通过结构、因果轨迹和在线误差门，$n=63$ 的通过比例都是 4095/4096，1% 分位余量
约为 $+0.20$。这说明剩余问题集中在极少数最坏随机点，不代表整体动力学重新失稳。根据用户
“不被极小误差带偏”的决策，第三阶段训练已停止，不再为形式上的全点通过继续消耗 RTX 2060。

## 结论边界

- 可以写入论文的实验事实：三传感器下，非线性条件可逆 $T_\phi$ 与 $B$ 的联合训练能修复
  原低模态坏点，并在 $31/63/127$ 三网格保持轨迹收缩、可逆性和在线性能。
- 不应写成：已经证明连续 Allen--Cahn PDE 在整个状态域全局收缩。
- seed 1851 已被用于挑选和 hard-point 回放，因此最终正余量是校准结果，不是独立样本外证据。
- seed 1871 和 locked test 1901 未读取；多 seed 第三阶段被主动中止且没有结果文件。

这条路线最有价值的博士课题结论不是“小数点后的最坏余量”，而是：在三传感器不足以给出
简单欧氏全局证书时，学习一个依赖 $(u,e)$、可逆且跨网格共享的误差几何，确实能把真实
Allen--Cahn 误差动力学从近似收缩推进到可用的有限域收缩。

## Validation Report

- Overall Confidence: MODERATE
- 工程复现：成功 checkpoint 的三网格门均通过；完整代码测试为 101 passed。
- 统计范围：没有总体显著性推断；数字只描述有限 collocation、有限轨迹和一个选定校准模型。
- 完整性：成功 `results.json` 的 SHA-256 为
  `850e49bf86c0a49f8308fb31c84fc0d2b60e865e67c6d0b0b8b287f294c0ccb6`；原始结果和 checkpoint
  保留在 RTX 2060。

### Fallacy Scan

- Coverage: 11/11 checked

| 类型 | 结论 | 本报告中的处理 |
|---|---|---|
| Simpson's paradox | NOTE | 三个网格分别报告，未用总体平均掩盖 $n=63$。 |
| Ecological fallacy | NOTE | 有限样本结论不外推到每个连续 PDE 状态。 |
| Berkson's paradox | CAUTION | 采样限制在预定低模态有限域。 |
| Collider bias | N/A | 未建立条件化因果模型。 |
| Base-rate neglect | NOTE | 同时报告最坏点和 4095/4096 通过率。 |
| Regression to mean | CAUTION | seed 1851 是校准数据，不称为独立验证。 |
| Survivorship bias | NOTE | 保留并报告多 seed 的失败点和中止阶段。 |
| Look-elsewhere effect | CAUTION | 选中 seed 1303 只作为工程 checkpoint，不包装为总体证书。 |
| Garden of forking paths | CAUTION | 所有校准修订按时间顺序记录，结论降级为探索性。 |
| Correlation != causation | NOTE | 容量瓶颈判断来自连续消融趋势，不称为数学证明。 |
| Reverse causality | N/A | 不涉及横截面因果方向推断。 |
