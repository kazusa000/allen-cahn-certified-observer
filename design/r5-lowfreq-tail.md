# R5：低频稳定与高频尾项路线

## 目标与边界

本路线对应目标库任务“验证低频稳定与高频尾项”。沿用研究计划中的

\[
\partial_tu=Au+F(u),\qquad
\partial_t\hat u=A\hat u+F(\hat u)+\Gamma_\theta(\hat u,y)(y-\obs\hat u),
\]

以及离线变换 (T_\phi)。不再要求一个全维 (T_\phi) 把所有 Allen--Cahn 误差统一
变成固定线性目标。

## 低频与尾部

令 (P_m) 为离散 Dirichlet Laplacian 前 (m=8) 个物理正弦模态的正交投影，
(Q_m=I-P_m)，并写

\[
p=P_me,\qquad q=Q_me,\qquad e=\hat u-u.
\]

三个网格都使用相同的物理模态编号 (1,\ldots,8)，而不是固定数组下标或固定比例。
低频 certificate 只在 (P_m) 内改变误差，尾部保持恒等：

\[
Q_mT_\phi(u,e)=q.
\]

同时继续保持

\[
T_\phi(u,0)=0,\qquad \obs T_\phi(u,e)=\obs e.
\]

实现上只在 (P_m\ker(\obs P_m)) 中学习可逆变换，因此不会靠改变高频尾部降低训练损失。
稳定损失、动力学缺陷和双向范数约束都只在 (P_m) 上计算；分母使用
(\lVert P_me\rVert_{M_h}^2+10^{-8})。在线校正默认也投影到 (P_m)，从结构上令
(Q_m\Gamma_\theta(\hat u,y)(y-\obs\hat u)=0)。

## 高频能量不等式

离散 Allen--Cahn 误差满足

\[
\partial_te=\nu L_he+e-\bigl((u+e)^3-u^3\bigr)+g_\theta.
\]

把三次项写成由 (p) 单独产生的低频到尾部耦合和含 (q) 的剩余项。由于
(v\mapsto v^3) 单调，含 (q) 的剩余项满足非正能量配对。若
(\mu_{m+1,h}) 是 (-L_h) 的第 (m+1) 个特征值，则样本上审计

\[
\frac12\frac{\mathrm d}{\mathrm dt}\lVert q\rVert_{M_h}^2
\le
-\bigl(\nu\mu_{m+1,h}-1\bigr)\lVert q\rVert_{M_h}^2
+\lVert q\rVert_{M_h}
\left(
\left\lVert Q_m\bigl((u+p)^3-u^3\bigr)\right\rVert_{M_h}
+\lVert Q_mg_\theta\rVert_{M_h}
\right).
\]

因此正式结果必须分别报告扩散裕量、低频到尾部耦合、校正的尾部注入、实际尾部能量变化和
不等式余量。沿保存时刻还要用每条轨迹上观测到的最大强迫递推 sampled envelope；该
envelope 只审计保存时刻，不外推为时间步之间的连续上界。固定增益和学习观测器必须分别
使用各自实际部署的校正项，不能混用轨迹与向量场，也不能把低频 loss 直接冒充全状态证书。

## 冻结比较与完成判据

- 低频维数：(m=8)。
- 参数、网格、训练/validation/test 划分、噪声和在线信息边界沿用
  `design/r5-formal-contract.md`。
- 对照：既有全状态联合训练、固定增益 nudging、低频投影校正。
- 本机只做单网格短 epoch smoke；多 seed、三网格正式训练放到 2060。
- 低频 (T_\phi) 必须通过零纤维、观测方向、Jacobian 奇异值和尾部恒等审计。
- 每个参数与网格报告低频缺陷、低频衰减、尾部能量比例、扩散裕量和耦合项。
- 只有低频稳定裕量与高频尾项控制能够合并时，才给出总误差结论；否则明确失败参数或网格。
