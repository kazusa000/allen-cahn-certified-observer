# R5-J：Sylvester 线性骨架与真实非线性余项

## 研究问题

R5-I 已经表明，当前学习坐标在样本上可以直接收缩，但把变换后的动力学强行写成一个
同形 Allen--Cahn 方程并不合理。本实验进一步回答：

> 只要求线性部分具有精确的目标动力学，并保留坐标变换后真实出现的非线性余项，
> 能否得到一个既精确、又在更宽状态分布上保持收缩的坐标？

本实验不训练新模型，也不读取 locked test。它是对线性 PDE 中 Sylvester/F-equivalence
构造在 Allen--Cahn 误差系统上的直接迁移测试。

## Material Passport

- 方程：一维 Allen--Cahn，齐次 Dirichlet 边界；
- 黏性：固定为 \(\nu=0.005\)；
- 观测：固定三个局部平均传感器；
- 网格：\(n\in\{31,63,127,191\}\)；
- 低模态：恰好四个线性不稳定模态；
- 数据来源：已有 IID collocation 生成器和 R5-H 的五类 OOD 生成器；
- 对照：冻结的 seed-1303 learned \(B+T_\phi\)，checkpoint 只读且哈希前后不变；
- 新构造：每个网格上确定性求解，不含可训练参数；
- 信息隔离：seed 2231 用于本实验 IID/OOD，seed 2232 用于对抗搜索；不读取 seed 1901。

## 精确线性结构

半离散误差动力学为

\[
\dot e=A_he-B_hC_he+N(u,e),\qquad
A_h=\nu\Delta_h+I,
\]

\[
N(u,e)=-[(u+e)^3-u^3].
\]

在离散正弦基中，把前四个不稳定模态记为 \(u\)，其余稳定模态记为 \(s\)。只平移不稳定
模态，稳定尾部保持原动力学：

\[
G_{u,h}=A_{u,h}-(1+\alpha)I,\qquad
G_h=\operatorname{diag}(G_{u,h},A_{s,h}),
\]

其中 \(\alpha=0.1\nu\pi^2\)。低模态上的约束 Sylvester 方程为

\[
(A_{u,h}-B_{u,h}C_{u,h})P_{u,h}=P_{u,h}G_{u,h},
\qquad C_{u,h}P_{u,h}=C_{u,h}.
\]

稳定尾部可能被传感器看见，因此再求交叉块

\[
(A_{u,h}-B_{u,h}C_{u,h})X_h-X_hA_{s,h}=B_{u,h}C_{s,h}.
\]

由此组成

\[
P_{m,h}=
\begin{bmatrix}P_{u,h}&X_h\\0&I\end{bmatrix},
\qquad T_h=P_h^{-1},
\]

并逐网格验证

\[
(A_h-B_hC_h)P_h=P_hG_h,\qquad
T_h(A_h-B_hC_h)=G_hT_h.
\]

不能把所有高频模态一起平移：三个局部传感器对部分高频模态不可观，而这些模态本来已经
稳定。只平移四个不稳定模态是与线性 PDE 构造真正对应的版本。

## 真实非线性余项

定义固定线性坐标

\[
z=T_he.
\]

则不再人为规定非线性目标，而是保留精确动力学

\[
\dot z=G_hz+R_h(u,z),
\qquad
R_h(u,z)=T_hN(u,P_hz).
\]

逐样本报告

\[
r_L=-\frac{\langle z,G_hz\rangle_h}{\|z\|_h^2},\qquad
r_N=-\frac{\langle z,R_h(u,z)\rangle_h}{\|z\|_h^2},
\]

\[
r=r_L+r_N,\qquad m=r-\alpha.
\]

同时验证由原误差动力学直接算出的 \(\dot z\) 与 \(G_hz+R_h\) 重建一致。

## 冻结比较

所有方法使用完全相同的样本：

1. Sylvester \(B_h+T_h\)：本实验的精确坐标；
2. Sylvester \(B_h+I\)：辨别收益来自注入还是坐标；
3. frozen learned \(B+T_\phi\)：已有直接收缩路线的只读对照。

正式 IID 每个网格 4096 点。OOD 覆盖 truth high frequency、error high frequency、
localized pulse、multiple interfaces、large initial error 五类，每类三个严重度，每格
512 点。OOD 是本实验的新审计，不改变此前已保留的 R5-G/R5-H 结论。

## 判据

### 结构门

- 低模态 Sylvester、完整相似变换、逆矩阵和非线性重建的相对误差均不超过 \(10^{-10}\)；
- 四个不稳定模态可观；
- \(\kappa_2(P_h)\le 10^5\)；
- 目标线性骨架的最慢衰减率不低于 \(\alpha\)。

### 实用门

门槛故意留有尾部空间。对 Sylvester 坐标，在每个网格的 IID 汇总和 OOD 汇总上分别要求：

- 非负收缩 margin 的比例至少 99%；
- margin 的 1% 分位数不小于 0；
- 所有结果有限。

最坏点仍完整报告，但单个孤立尾点不单独否决实用泛化。

### 证明候选门

只有结构门和实用门都通过后，才在预设紧致域上运行 64 个固定重启、100 步投影梯度搜索。
若每个网格找到的最坏 margin 仍严格大于 0，才称为“证明候选”；这仍不是连续 PDE 定理。

紧致域固定为：状态由前八个正弦模态表示且物理最大幅值不超过 1.25；误差由前十二个模态
表示且质量范数在 \([0.02,0.8]\) 内。

## 顺序决策

1. 若结构门失败：停止，不做统计解释；
2. 若结构门通过而实用门失败：保留当前直接收缩路线，把 Sylvester 构造作为失败对照；
3. 若实用门通过但对抗门失败：下一步才考虑以 \(T_h\) 为锚点的小型可逆非线性修正；
4. 若三门都通过：停止训练，转向网格一致和连续 PDE 的余项上界推导。

## 输出

正式结果写入新的 out/r5-j-* 目录，包含 JSON 原始指标与运行日志；整理结论写入 report/。
不得覆盖已有输出，不得修改冻结 checkpoint，不得更新父仓库 submodule 指针。
