# R5：$T_0$ 预条件非线性变换联合训练计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-24
- Verification Status: VERIFIED — FORMAL VALIDATION FAILED (NONLINEARITY GATE ONLY)
- Version Label: code_plan_v1

## 研究问题

在

\[
\nu=0.005,\qquad q=3,\qquad n=31
\]

下，上一轮以恒等映射为中心的条件可逆残差变换丢失了固定 LMI 变换 $T_0$ 在前四个不稳定
模态上的各向异性几何。本轮检验：若把 $T_0$ 作为不可训练的全局预条件骨架，只学习其上的
非线性修正，能否在保留可逆性和在线性能的同时，使实际变换动力学满足非线性目标缺陷和
最坏收缩门？

本轮仍只回答固定半离散网格 $n=31$ 的有限样本 validation 问题，不声称连续 PDE 定理。

## 固定模型与目标

沿用

\[
\partial_tu=Au+F(u),
\qquad
\partial_t\hat u=A\hat u+F(\hat u)+B(y-\mathcal C\hat u),
\]

\[
A=\nu\Delta_h,qquad F(v)=v-v^3,qquad e=\hat u-u.
\]

三个传感器仍固定为

\[
[1/6,7/30],\qquad[7/15,8/15],\qquad[23/30,5/6].
\]

本轮不采用频带投影目标，继续使用完整非线性目标

\[
G_\lambda(u,z)
=Az+F(u+z)-F(u)-(1+\lambda)z,
\qquad
\lambda=0.1\nu\pi^2.
\]

原因是已完成的四模态审计表明：约 95% 的目标缺陷和约 92% 的收缩失败负功率负担来自
$\Pi_4$，把阻尼改成 $-(1+\lambda)\Pi_4z$ 对现有 checkpoint 的缺陷 RMS 仅改善约 2.1%。

## $T_0$ 预条件条件可逆变换

令 $T_0$ 为同一传感器几何、同一 $\nu$ 和同一网格上固定 LMI 度量得到的归一化线性变换。
本轮只使用

\[
T_\phi(u,e)
=T_0\left[e+\tilde g_\phi(u,e)-\tilde g_\phi(u,0)\right].
\]

等价地，若

\[
g_\phi(u,e)=T_0\tilde g_\phi(u,e),
\]

则

\[
T_\phi(u,e)=T_0e+g_\phi(u,e)-g_\phi(u,0).
\]

该结构自动满足 $T_\phi(u,0)=0$，并且当非线性修正为零时精确退化为 $T_0e$，不会像上一轮
那样把固定稳定几何排除在模型类之外。

$\tilde g_\phi$ 继续使用三层、宽度 128 的带 $u$ 加性条件输入的 $\tanh$ 网络。固定

\[
\rho=0.5,
\qquad
2\sup_{u,e}\|D_e\tilde g_\phi(u,e)\|_2\leq\rho.
\]

因此

\[
S_\phi(u,e)=e+\tilde g_\phi(u,e)-\tilde g_\phi(u,0)
\]

关于 $e$ 全局可逆，且

\[
T_\phi(u,\cdot)=T_0\circ S_\phi(u,\cdot)
\]

也全局可逆。它等价于

\[
\sup_{u,e}\|T_0^{-1}D_eg_\phi(u,e)\|_2\leq\rho/2<1.
\]

若 $s_{\min}(T_0),s_{\max}(T_0)$ 是 $T_0$ 的极端奇异值，则全局保证

\[
s_{\min}(T_0)(1-\rho/2)
\leq\sigma_{\min}(D_eT_\phi),
\]

\[
\sigma_{\max}(D_eT_\phi)
\leq s_{\max}(T_0)(1+\rho/2).
\]

数值逆先计算 $T_0^{-1}z$，再对 $S_\phi$ 做固定点迭代，不学习独立逆网络。

## 动力学和四项损失

实际变换动力学仍使用完整链式法则

\[
\partial_tz_{\mathrm{act}}
=D_uT_\phi(u,e)\partial_tu+D_eT_\phi(u,e)\partial_te.
\]

损失仍严格只有四项：

\[
\mathcal L
=\mathcal L_{\mathrm{动力学收缩}}
+\mathcal L_{\mathrm{非线性目标缺陷}}
+\mathcal L_{\mathrm{可逆性}}
+\mathcal L_{\mathrm{在线误差}}.
\]

四项定义、归一化、等权求和、最坏 10% 收缩 CVaR、在线 RK4 rollout、梯度裁剪和当前
$B$ 轨迹每 20 epoch 刷新均与上一轮完全相同。$B$ 从相同 LMI $B_0$ 初始化并执行硬投影

\[
\|B-B_0\|_F\leq0.25\|B_0\|_F.
\]

不增加第五个损失，不加入 $\Pi_4$ 专门损失，也不改变目标方程。

## 新数据划分

上一轮 validation seed 801--802 已用于模型诊断和频带归因，不能继续作为新结构的无泄漏
validation。本轮生成完全新的确定性划分：

- train case seeds：711、712、713、714，共 16 条基础轨迹；
- validation case seeds：811、812，共 8 条基础轨迹；
- locked test case seeds：911、912，共 8 条基础轨迹；
- 每个 split 仍从前两条基础真值构造第四模态和最小观测方向的正负困难轨迹 8 条；
- 因此 train、validation、locked test 各有 24、16、16 条轨迹；
- 模型 seed：1201、1202、1203。

test 仍只有至少 2/3 模型 seed 通过全部 validation 门后才执行一次。

## 结构与非线性审计

除零纤维、谱乘积、数值逆和物理 Jacobian 奇异值外，同时在归一化坐标
$S_\phi=T_0^{-1}T_\phi$ 中报告

\[
\tilde\eta_e
=\frac{\mathbb E\|S_\phi(u,2e)-2S_\phi(u,e)\|_h}
{\mathbb E\|S_\phi(u,e)\|_h+\varepsilon},
\]

\[
\tilde\eta_u
=\frac{\mathbb E\|S_\phi(u_1,e)-S_\phi(u_2,e)\|_h}
{\mathbb E\|S_\phi(u_1,e)\|_h+\varepsilon}.
\]

由于 $T_0$ 是固定可逆线性映射，$T_\phi$ 关于 $e$ 是否非线性与 $S_\phi$ 是否非线性等价。
冻结通过门为

\[
\tilde\eta_e\geq0.02,
\qquad
\tilde\eta_u\geq0.02.
\]

同时继续报告原物理 $z$ 坐标中的 $\eta_e,\eta_u$，但不重复设置第二套数值门。

## 训练配置

- 三个 seed 从第一个 epoch 起联合训练 $B,T_\phi$；
- 每个 seed 80 epoch；
- instant batch 512，rollout batch 4；
- $B$ 学习率 $5\times10^{-4}$，$T_\phi$ 学习率 $10^{-3}$；
- hidden width 128，hidden layers 3，$\rho=0.5$；
- 全局梯度裁剪 1.0；
- RTX 2060 正式运行硬超时 12 小时；
- 不根据中间 seed 修改参数。

## Validation 成功门

每个 seed 必须同时满足：

1. 数值有限且 $T_\phi(u,0)=0$；
2. $2\sup\|D_e\tilde g_\phi\|_2\leq0.5$；
3. 采样物理 Jacobian 位于上述 $T_0$ 缩放全局界，固定点逆相对误差不超过 $10^{-6}$；
4. $\tilde\eta_e\geq0.02$ 且 $\tilde\eta_u\geq0.02$；
5. 全部 validation 保存点满足要求收缩率，最坏裕量非负；
6. 相对新 validation 上固定 $B_0,T_0$，目标缺陷 RMS 至少降低 25%，95% 分位至少降低
   15%；
7. 在线终点误差中位数不超过固定 $B_0$ 的 1.05 倍，最大值不超过 1.10 倍；
8. 至少 2/3 模型 seed 独立通过第 1--7 项。

门槛不因上一轮结果放宽。若 validation 总门失败，locked test 不执行。

## 对照和解释规则

固定对照仍为 $B_0,T_0$ 与 $B_0,I$。本轮不重新训练恒等映射中心模型，因为上一轮已给出
其失败证据，且数据划分已经更换。

- 若收缩明显恢复但缺陷改善不足：说明 $T_0$ 几何有效，但非线性修正尚未完成动力学匹配；
- 若缺陷和收缩通过但 $\tilde\eta_e$ 失败：结果主要来自固定线性 $T_0$，不能支持非线性
  变换增量价值；
- 若在线通过而动力学失败：只支持 $B$ 的在线改进，不支持动力学收缩；
- 若至少 2/3 seed 全部通过：只称为新 validation 上的 $n=31$ 数值证据，然后解锁一次
  locked test。

## 计划产物

| 产物 | 位置 |
|---|---|
| 预条件网络 | `src/allen_cahn_certified_observer/nonlinear_target.py` |
| 训练入口 | `tool/r5_nonlinear_target_T0_preconditioned_joint.py` |
| 正式测试 | `tests/test_nonlinear_target_T0_preconditioned_joint.py` |
| checkpoint | `out/<new>/checkpoints/` |
| 正式结果 | `out/<new>/results.json` |
| 结果报告 | `report/r5-nonlinear-target-T0-preconditioned-joint-20260824.md` |
