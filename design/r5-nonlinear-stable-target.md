# R5：保留非线性的稳定目标

## Material Passport

- ID：R5-nonlinear-stable-target-2026-08-24
- 类型：Code Experiment Plan
- 状态：COMPLETED — PROGRESS GATE FAILED
- 前序结果：`report/r5-dynamics-defect-repair.md`

## 研究问题

研究计划从

\[
\partial_t u=Au+F(u),\qquad y=\obs u
\]

出发，并用 $z=T_\phi(u,e)$ 把重构误差连接到稳定目标。当前 R5 强制所有状态和误差都满足

\[
\partial_t z=(A-\lambda I)z,
\]

但独立验证动力学缺陷 RMS 仍为 0.6566。要检验的问题是：失败是否来自固定线性目标丢掉了
Allen--Cahn 非线性增量，而不是 $T_\phi$ 和 $B$ 的容量不足。

## 状态相关目标

主目标保留完整非线性增量：

\[
\partial_t z
=Az+F(u+z)-F(u)-(1+\lambda)z.
\]

对 Allen--Cahn 反应项 $F(v)=v-v^3$，有

\[
\left\langle F(u+z)-F(u),z\right\rangle
\leq \|z\|^2.
\]

因此

\[
\frac12\frac{\mathrm d}{\mathrm dt}\|z\|^2
\leq \langle Az,z\rangle-\lambda\|z\|^2.
\]

这和旧目标 $Az-\lambda z$ 具有相同的保守衰减下界，同时保留了随 $u$ 和 $z$ 变化的
非线性结构。阻尼中的 1 不是调参结果，而是补偿 $F(v)=v-v^3$ 的最大单边增长率。

同时设置局部线性化对照：

\[
\partial_t z=Az+DF(u)z-(1+\lambda)z.
\]

它保留状态依赖，但不保留有限误差下的高阶项，用于区分“状态依赖”与“完整非线性增量”
各自的作用。

## 离散实现

连续动力学缺陷直接使用上述右端。一步稳定目标使用扩散半步、冻结在相邻真状态中点的反应
步、扩散半步；反应步用四阶 Runge--Kutta。旧线性目标继续使用矩阵指数。三种目标共用：

- 相同训练、验证和测试轨迹；
- 相同 $T_\phi$ 三角可逆结构；
- 相同常数正增益 $B$ 及其约束；
- 相同 $\lambda$、损失权重、epoch、seed 和模型选择规则。

因此目标类型是第一阶段唯一的科学变量。

## 预注册筛选

第一阶段固定 `n=31`、seed 501--503、80 epoch，每 20 epoch 刷新当前观测器轨迹，比较：

1. 旧固定线性目标；
2. 保留 $DF(u)$ 的目标；
3. 保留 $F(u+z)-F(u)$ 的目标。

每种目标先通过 $T_\phi(u,0)=0$、观测方向和 Jacobian 奇异值审计，再要求 validation
在线误差不差于固定增益 0.10，最后按固定增益与当前观测器两类独立 validation 轨迹中较大
的动力学缺陷 RMS 选择 seed。

进入多网格正式复核需要同时满足：

- 相对同轮线性目标，验证动力学缺陷 RMS 至少下降 25%；
- 验证动力学缺陷 RMS 不超过前序冻结绝对门槛 0.3668；
- validation 在线误差不退化且可逆性约束通过。

局部稳定结论还要求每个 Allen--Cahn 参数组的验证 RMS 小于由目标衰减和 Jacobian 下界给出
的保守门槛。若第一阶段没有通过进展门槛，则保留负结果，不用扩大训练规模掩盖失败。

## 执行结果

正式筛选已在 RTX 2060 上完成。保留 $DF(u)$ 和保留完整非线性增量的目标均通过可逆性与
在线误差约束，但验证动力学缺陷分别比同轮线性目标增加 14.5% 和 15.8%，相对与绝对进展
门槛均失败。因此按预注册规则停止在 `n=31`，不进入多网格扩展。完整结果见
`report/r5-nonlinear-stable-target.md`。
