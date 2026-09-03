# R5-I：backstepping 非线性余项诊断

## 研究问题

冻结的三传感器直接收缩观测器已经在独立样本、未训练网格和多数状态分布外测试中保持
正收缩。本实验不重新训练，而是回答：此前指定的 Allen--Cahn 非线性目标为何难以精确匹配，
以及其缺陷是否主要是不影响能量下降的切向分量。

误差动力学为

\[
\partial_t e=\nu\Delta_he+e-[(u+e)^3-u^3]-B\mathcal Ce,
\]

学习坐标为 \(z=T_\phi(u,e)\)。旧的同形非线性目标是

\[
G_{\rm same}(u,z)
=\nu\Delta_hz-[(u+z)^3-u^3]-\alpha z,
\qquad \alpha=0.1\nu\pi^2.
\]

目标缺陷定义为

\[
d=\partial_tz-G_{\rm same}(u,z).
\]

## 缺陷分解

用完整链式法则把缺陷精确拆成

\[
d=d_u+d_\Delta+d_{B}+d_N,
\]

其中

\[
\begin{aligned}
d_u&=D_uT_\phi\,\partial_tu,\\
d_\Delta&=D_eT_\phi(\nu\Delta_he)-\nu\Delta_hz,\\
d_B&=D_eT_\phi(e-B\mathcal Ce)+\alpha z,\\
d_N&=D_eT_\phi\bigl(-[(u+e)^3-u^3]\bigr)
      +[(u+z)^3-u^3].
\end{aligned}
\]

这四项的和必须逐样本重建总缺陷。另将总缺陷正交分解为

\[
d_\parallel=\frac{\langle z,d\rangle_h}{\|z\|_h^2}z,
\qquad d_\perp=d-d_\parallel.
\]

\(d_\parallel\) 改变能量衰减率，\(d_\perp\) 只改变瞬时运动方向。

## 冻结协议

- \(\nu=0.005\)，固定三个局部平均传感器；
- checkpoint：R5-G/R5-H 使用的冻结 seed-1303 模型，运行前后 SHA-256 必须一致；
- audit seed：2111，不用于训练或调参；
- 网格：\(n\in\{31,63,127,191\}\)，其中 \(n=191\) 未参与训练；
- 每个网格 4096 个 collocation 样本；
- 比较三种模型：learned \(B+T_\phi\)、learned \(B\)+fixed \(T_0\)、baseline \(B_0+T_0\)；
- 全程 evaluation-only，不创建优化器、不反向传播、不写回 checkpoint；
- locked test seed 1901 不再读取。

## 预注册诊断判据

1. **实现完整性**：四项缺陷重建的最大相对误差不超过 \(10^{-8}\)，所有数字有限，
   checkpoint 哈希不变。
2. **旧目标近似成立**：只有每个网格的归一化缺陷 RMS 均不超过 0.1，才称旧目标在当前域内
   得到近似实现。
3. **切向主导**：只有每个网格 \(d_\perp\) 占缺陷平方能量至少 75%，且池化占比至少 80%，
   才把“增加反对称旋转项”升级为首选后续路线。
4. **结构来源**：分别报告四项的归一化 RMS 与对收缩率的影响；分量范数不可相加，因此
   不用单项 RMS 百分比冒充严格因果归因。
5. **训练决策**：若旧目标不近似成立但切向主导，则下一阶段只学习受约束旋转项；若径向
   缺陷明显，则不训练旋转项，改为线性 backstepping 骨架加真实变换后非线性余项的稳定性
   分析。当前审计本身不训练新模型。

## 顺序触发的余项分析

第一阶段正式结果选择了“线性骨架加真实非线性余项”路线。以下分析是在看到该路线选择后、
正式汇总前增加的确定性重组，不作为预注册的独立验证，也不增加新门槛。对 fixed (T_0)
坐标，定义网格共享的线性变换 (T_{0,h})，则动力学可精确写为

\[
\partial_t z
=\underbrace{T_{0,h}(\nu\Delta_he+e-B\mathcal Ce)}_{\text{线性闭环骨架}}
+\underbrace{T_{0,h}\!\left(-[(u+e)^3-u^3]\right)}_{\text{真实非线性余项}},
\qquad e=T_{0,h}^{-1}z.
\]

报告两部分各自的能量衰减率、线性骨架达到请求速率的样本比例、非线性余项具有非负耗散率
的样本比例，并验证各部分衰减率逐样本精确相加。learned (T_\phi) 同时报告
(D_uT_\phi\,\partial_tu)，但只把 fixed (T_0) 的第一项称为严格意义的线性骨架。

## 结论边界

本实验只能判断冻结模型和规定有限样本域上的目标缺陷结构。它不证明非线性共轭存在或
不存在，也不把半离散样本结果外推为连续 PDE 定理。
