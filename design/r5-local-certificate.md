# R5：分区局部稳定证书

## Status

本设计在查看分区结果前冻结。它只审计已经选定的 R5 观测器与
\(T_\phi\)，不重新选择 checkpoint，也不根据 validation 结果调整阈值。

## Mathematical audit

沿当前在线观测器轨迹令

\[
e=\hat u-u,\qquad z=T_\phi(u,e),
\]

并记

\[
r_T=\partial_uT_\phi(u,e)\,\partial_tu
    +\partial_eT_\phi(u,e)\,\partial_te
    -(A-\lambda I)z.
\]

每个样本直接记录以下归一化动力学缺陷和实际变换后收缩率

\[
\frac{
\left\lVert
\partial_uT_\phi(u,e)\,\partial_tu
+\partial_eT_\phi(u,e)\,\partial_te
-(A-\lambda I)T_\phi(u,e)
\right\rVert_{M_h}}
{\lVert e\rVert_{M_h}},
\qquad
-\frac{\langle z,\partial_tz\rangle_{M_h}}
      {\lVert z\rVert_{M_h}^2}.
\]

在每个区域中，取 \(A-\lambda I\) 的最慢衰减率、上式归一化缺陷的最大值，以及
\(T_\phi\) 对 \(e\) 的 Jacobian 最小奇异值。用“最慢衰减率减去最大归一化缺陷除以
最小奇异值”作为严格有限样本裕量。只有样本数不少于 30、全部数值有限、观测方向与
零误差纤维约束通过且该裕量为正时，该区域才记为有限样本通过。RMS 缺陷只作为诊断，
不替代最坏样本判据。

## Frozen physical partition

状态主分区互斥且覆盖全部样本，按以下优先级定义：

1. `zero-near`：\(\lVert u\rVert_{M_h}\le 0.25\)；
2. `phase-dominated`：至少 50% 的内部节点满足 \(|u_j|\ge0.75\)；
3. `interface-mixed`：其余状态。

误差尺度使用研究计划中已声明的 \(0.05\)--\(0.25\) 初始误差范围：

- `small`：\(\lVert e\rVert_{M_h}\le0.05\)；
- `medium`：\(0.05<\lVert e\rVert_{M_h}\le0.15\)；
- `large`：\(\lVert e\rVert_{M_h}>0.15\)。

参数分区保持 \(\nu\in\{0.005,0.010,0.020\}\) 不合并。正式表同时报告三个
状态主区、三个误差区、三个参数组以及样本数不少于 30 的交叉区域。

为检查局部证书之间的重叠，另定义扩张邻域：`zero-near` 使用半径 0.30，
`phase-dominated` 使用节点比例 0.40，`interface-mixed` 使用
\(\lVert u\rVert_{M_h}\ge0.20\) 且相态节点比例不超过 0.60。扩张邻域仅用于覆盖和
重叠审计，不用于改变主分区。

## Frozen data and comparisons

- 网格：先在 \(n=31\) 审计；只有发现通过区域或明确需要跨网格确认边界时才扩展。
- 数据：固定的 R5 validation split，使用当前在线观测器的因果轨迹。
- checkpoint：R5 动力学缺陷修复筛选中预先选定的
  `triangular-current-policy/grid-31__seed-501.pt`。
- 基线：同一 validation case 上的固定增益 0.10 观测器。
- 每个区域报告：覆盖率、动力学缺陷、实际收缩率、Jacobian 奇异值、在线误差和
  相对固定增益误差。
- 轨迹切换报告：主状态区切换次数、切换前后样本以及扩张邻域覆盖。当前首轮使用同一个
  \(T_\phi\) 在各区域限制审计，因此坐标切换跳跃严格为零；后续若改成多个局部
  \(T_\phi\)，必须额外报告重叠区映射差异。

## Decision rule

- 若至少一个预先声明区域满足严格正裕量，则得到该区域内的有限样本局部证书候选，随后
  扩大独立样本并做跨网格复核。
- 若没有区域通过，则报告各区域失败来源；不得通过缩小区域到单个低缺陷样本或重新选择
  checkpoint 制造正结果。
