# R5：非线性 Allen--Cahn 目标与条件可逆残差变换联合训练计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-24
- Verification Status: VERIFIED — FORMAL RUN COMPLETE; VALIDATION FAILED
- Version Label: executed_plan_v1

## 研究问题

在

\[
\nu=0.005,\qquad q=3,\qquad n=31
\]

的欠通道条件下，能否联合学习观测注入 $B$ 和真正非线性的条件可逆变换
$T_\phi(u,e)$，使实际 Allen--Cahn 观测误差在变换坐标中逼近一个保留原方程非线性结构、
并具有严格收缩性质的目标动力学，同时不牺牲在线重构误差？

本轮只回答固定半离散网格 $n=31$ 上的 validation 可行性，不进行多网格外推，也不声称
连续 PDE 定理。

## 核心假设

上一轮线性目标

\[
\partial_tz=(A-\lambda I)z
\]

要求学习变换在较大的非线性状态区域内消除 Allen--Cahn 三次反应结构，目标过强。本轮改为
保留该结构的非线性收缩目标：

\[
\partial_tz
=Az+F(u+z)-F(u)-(1+\lambda)z.
\]

研究假设是：条件可逆、关于 $e$ 真正非线性的 $T_\phi(u,e)$，比原来的 $P(u)e$
更有能力匹配这一目标；联合训练 $B,T_\phi$ 后，可在全部 validation 困难轨迹上同时取得
非负收缩裕量、较小的非线性目标缺陷和不劣于固定 LMI 参考的在线误差。

## 数学模型

沿用研究计划中的对象

\[
\partial_tu=Au+F(u),\qquad y=\mathcal C u,
\]

\[
\partial_t\hat u=A\hat u+F(\hat u)+B(y-\mathcal C\hat u).
\]

固定 $e=\hat u-u$，则由观测器直接得到

\[
\partial_te
=Ae+F(u+e)-F(u)-B\mathcal C e.
\]

本轮明确采用

\[
A=\nu\Delta_h,\qquad F(v)=v-v^3,
\]

定义在 $[0,1]$ 上，使用齐次 Dirichlet 边界、$n=31$ 个内部网格点和
$M_h=hI$。实现前必须检查代码中的 $A$ 不包含反应项 $+I$；否则不得直接使用本计划的
$(1+\lambda)z$ 系数。

三个局部平均传感器固定为

\[
[1/6,7/30],\qquad[7/15,8/15],\qquad[23/30,5/6],
\]

总观测长度为 $0.20$。收缩率固定为

\[
\lambda=0.1\nu\pi^2\approx0.0049348.
\]

## 非线性目标为何收缩

定义

\[
G_\lambda(u,z)
=Az+F(u+z)-F(u)-(1+\lambda)z.
\]

由于 $F(v)=v-v^3$，有

\[
F(u+z)-F(u)-(1+\lambda)z
=-\bigl((u+z)^3-u^3\bigr)-\lambda z.
\]

因此在离散质量内积下

\[
\begin{aligned}
\frac12\frac{\mathrm d}{\mathrm dt}\|z\|_h^2
&=\langle z,Az\rangle_h
-\left\langle z,(u+z)^3-u^3\right\rangle_h
-\lambda\|z\|_h^2\\
&=\langle z,Az\rangle_h
-h\sum_i z_i^2\left(z_i^2+3u_i z_i+3u_i^2\right)
-\lambda\|z\|_h^2\\
&\leq-\lambda\|z\|_h^2,
\end{aligned}
\]

因为

\[
z_i^2+3u_i z_i+3u_i^2
=\left(z_i+\frac32u_i\right)^2+\frac34u_i^2\geq0
\]

且 $\langle z,Az\rangle_h\leq0$。所以目标系统本身保留 Allen--Cahn 非线性，并在
$M_h$ 范数下至少以 $\lambda$ 收缩。

## 条件可逆残差网络

只使用

\[
T_\phi(u,e)=e+g_\phi(u,e)-g_\phi(u,0).
\]

该结构自动满足

\[
T_\phi(u,0)=0.
\]

$g_\phi$ 使用三层、宽度 128 的 $\tanh$ 网络；$u$ 在每一隐藏层以加性条件输入进入：

\[
h_1=\tanh(W_1e+U_1u+b_1),
\]

\[
h_{\ell+1}=\tanh(W_{\ell+1}h_\ell+U_{\ell+1}u+b_{\ell+1}),
\qquad
g_\phi(u,e)=W_oh_3+b_o.
\]

所有从 $e$ 到输出的权重使用谱归一化；$\tanh$ 为 1-Lipschitz。固定

\[
\rho=0.5,
\qquad
2\sup_{u,e}\|D_eg_\phi(u,e)\|_2\leq\rho<1.
\]

差分项 $g_\phi(u,0)$ 对 $e$ 的导数为 0，所以保证全局可逆实际只需要
$\sup\|D_eg_\phi\|_2<1$。本轮保留因子 2 作为更严格的安全余量。它给出更紧的界

\[
1-\frac\rho2
\leq\sigma_{\min}(D_eT_\phi)
\leq\sigma_{\max}(D_eT_\phi)
\leq1+\frac\rho2,
\]

从而也必然满足用户提出的较松声明

\[
1-\rho
\leq\sigma_{\min}(D_eT_\phi)
\leq\sigma_{\max}(D_eT_\phi)
\leq1+\rho.
\]

因为 $e\mapsto g_\phi(u,e)-g_\phi(u,0)$ 的 Lipschitz 常数严格小于 1，给定任意 $u,z$，
方程

\[
e=z-g_\phi(u,e)+g_\phi(u,0)
\]

是压缩不动点问题。因此 $T_\phi(u,\cdot)$ 关于 $e$ 全局可逆；数值逆通过固定点迭代
计算，不额外学习逆网络。

## 排除退化为 \(P(u)e\)

使用 $\tanh$ 只代表网络具有非线性能力，不能保证训练后没有停留在近线性区域。因此以下
指标只做 validation 硬审计，不增加损失项：

\[
\eta_e=
\frac{\mathbb E\|T_\phi(u,2e)-2T_\phi(u,e)\|_h}
{\mathbb E\|T_\phi(u,e)\|_h+\varepsilon},
\]

\[
\eta_u=
\frac{\mathbb E\|T_\phi(u_1,e)-T_\phi(u_2,e)\|_h}
{\mathbb E\|T_\phi(u_1,e)\|_h+\varepsilon}.
\]

冻结通过门为

\[
\eta_e\geq0.02,
\qquad
\eta_u\geq0.02.
\]

同时报告 $\|D_{ee}^2T_\phi\|$ 的随机方向有限差分统计。若两项主指标任一不通过，即使其他
结果良好，也只能称为近线性变换，不能用于证明“非线性 $T_\phi$ 有价值”。

## 实际变换动力学与目标缺陷

令

\[
z=T_\phi(u,e).
\]

训练和验证必须使用完整链式法则

\[
\partial_tz_{\mathrm{act}}
=D_uT_\phi(u,e)\bigl(Au+F(u)\bigr)
+D_eT_\phi(u,e)
\bigl(Ae+F(u+e)-F(u)-B\mathcal Ce\bigr).
\]

非线性目标缺陷定义为

\[
r_\phi(u,e)
=\partial_tz_{\mathrm{act}}-G_\lambda(u,z).
\]

任何只使用 $D_eT_\phi\,\partial_te$、遗漏 $D_uT_\phi\,\partial_tu$ 的实现均视为数学
错误，不能进入正式训练。

## 联合训练变量

- $B\in\mathbb R^{31\times3}$：可部署的常值观测注入，从同传感器几何的 LMI $B_0$
  初始化；
- $T_\phi(u,e)$：上述条件可逆残差网络，从接近恒等映射初始化；
- $B,T_\phi$ 从第一个 epoch 起同时更新，不做只训练 $B$ 或只训练 $T_\phi$ 的训练
  分支；
- 固定 LMI $B_0,T_0$ 和 $B_0,T=I$ 只作不训练参照。

为防止 $B$ 通过无限增益降低短期误差，本轮不增加第五个增益正则损失，而在每次优化后做
硬投影：

\[
\|B-B_0\|_F\leq0.25\|B_0\|_F.
\]

## 四项损失

四项全部无量纲化后等权相加：

\[
\mathcal L
=\mathcal L_{\mathrm{动力学收缩}}
+\mathcal L_{\mathrm{非线性目标缺陷}}
+\mathcal L_{\mathrm{可逆性}}
+\mathcal L_{\mathrm{在线误差}}.
\]

### 1. 动力学收缩

定义实际收缩裕量

\[
m_\phi(u,e)
=-\frac{\langle z,\partial_tz_{\mathrm{act}}\rangle_h}
{\|z\|_h^2+\varepsilon}
-\lambda.
\]

损失不取全批次普通平均，而取最坏 10% 样本的均值：

\[
\mathcal L_{\mathrm{动力学收缩}}
=\operatorname{CVaR}_{10\%}
\left[\operatorname{ReLU}\bigl(-m_\phi(u,e)\bigr)^2\right].
\]

### 2. 非线性目标缺陷

\[
\mathcal L_{\mathrm{非线性目标缺陷}}
=\mathbb E
\frac{\|r_\phi(u,e)\|_h^2}
{\|e\|_h^2+\varepsilon}.
\]

正式结果同时报告 RMS、中位数、95% 分位和最大值。

### 3. 可逆性

谱归一化提供全局理论界；损失只检查数值逆和实际条件数：

\[
\begin{aligned}
\mathcal L_{\mathrm{可逆性}}
=\mathbb E\biggl[
&\frac{\|T_\phi^{-1}(u,T_\phi(u,e))-e\|_h^2}
{\|e\|_h^2+\varepsilon}\\
&+\operatorname{ReLU}\left(1-\frac\rho2-\sigma_{\min}(D_eT_\phi)\right)^2\\
&+\operatorname{ReLU}\left(\sigma_{\max}(D_eT_\phi)-1-\frac\rho2\right)^2
\biggr].
\end{aligned}
\]

固定点逆最多迭代 50 次，停止阈值为 $10^{-8}$。未收敛样本直接记为结构失败，不用有限
损失掩盖。

### 4. 在线误差

在因果观测器 rollout 上定义

\[
\mathcal L_{\mathrm{在线误差}}
=\frac12\mathbb E
\left[
\frac1K\sum_{k=1}^{K}
\frac{\|e(t_k)\|_h^2}{\|e(0)\|_h^2+\varepsilon}
\right]
+\frac12\mathbb E
\left[
\frac{\|e(T)\|_h^2}{\|e(0)\|_h^2+\varepsilon}
\right].
\]

训练通过可微分时间积分对 $B$ 反向传播。$T_\phi$ 不进入在线观测器，因此该项不会直接
更新 $T_\phi$；这是有意设计，避免把依赖真实 $u,e$ 的证书变换误当作可部署模块。

## 数据、划分与困难样本

- 时间范围：$T=1.0$，保存 101 个均匀时刻；
- 初始真状态：前三个正弦模态，系数绝对值不超过 0.5；
- 初始误差：基础随机模态加第 4 不稳定模态和接近最小观测方向的双符号困难误差；
- train：新 seed 701--704，每个 4 个独立 draw，共 16 条基础轨迹并加入困难轨迹；
- validation：新 seed 801--802，每个 4 个独立 draw，共 8 条基础轨迹并加入不相交困难轨迹；
- locked test：seed 901--902，只在全部 validation 门通过后运行一次；
- 旧实验使用过的 validation seed 不参与本轮模型选择，避免路线迭代造成隐性验证集泄漏；
- 训练轨迹每 20 epoch 使用当前 $B$ 刷新一次；刷新不改变初值、真轨迹或数据划分。

## 训练设置

- 模型 seed：1101、1102、1103；
- 每个 seed：80 epoch；
- 瞬时动力学 batch size：512；
- 可微在线 rollout batch size：4；
- $B$ 学习率：$5\times10^{-4}$；
- $T_\phi$ 学习率：$10^{-3}$；
- 优化器：Adam；
- 梯度裁剪：全局范数 1.0；
- 精度：float32 训练，关键 validation Jacobian 和缺陷统计使用 float64 重算；
- 设备：2060 RTX 2060 6 GB；
- 正式运行硬超时：12 小时；
- 不因某个 seed 的中间结果修改权重、epoch、$\rho$ 或数据。

每个 seed 保存独立 checkpoint。模型选择顺序为：

1. 所有 validation 样本的最坏收缩裕量非负；
2. 全部结构与非线性审计通过；
3. 在线终点误差比例更小；
4. 非线性目标缺陷 RMS 更小。

## Validation 成功门

选中模型必须同时满足：

1. 数值有限，$T_\phi(u,0)=0$；
2. 谱范数乘积的全局界满足
   $2\sup\|D_eg_\phi\|_2\leq\rho=0.5$；
3. 所有 validation Jacobian 奇异值位于 [0.75,1.25]，固定点逆全部收敛且相对重构误差
   不超过 $10^{-6}$；
4. $\eta_e\geq0.02$ 且 $\eta_u\geq0.02$，排除 $P(u)e$ 或与 $u$ 无关的退化解；
5. 全部 validation 采样点满足

   \[
   -\frac{\langle z,\partial_tz_{\mathrm{act}}\rangle_h}
   {\|z\|_h^2+\varepsilon}\geq\lambda;
   \]

6. 非线性目标缺陷 RMS 相对固定 LMI $B_0,T_0$ 至少降低 25%，95% 分位至少降低 15%；
7. validation 在线终点误差中位数不超过固定 LMI $B_0$ 的 1.05 倍，最大误差不超过其
   1.10 倍；
8. 至少 2/3 个模型 seed 独立满足第 1--7 项，避免只报告偶然成功的单 seed。

只有这些门全部通过，才解锁一次 locked test。test 不参与调参，失败后不在同一计划内修改
模型再看 test。

## 对照与主要报告量

本轮只训练联合模型，不训练消融模型。两个固定对照为：

- 固定 LMI $B_0,T_0$：衡量新非线性目标缺陷和直接收缩是否优于原安全骨架；
- 固定 LMI $B_0$、$T=I$：显示条件变换相对原误差坐标的价值。

主要报告：

- 最坏收缩裕量、5% 分位裕量和收缩通过率；
- 非线性目标缺陷 RMS、95% 分位和最大值；
- 在线时间平均、峰值、终点误差及其相对固定 LMI 比例；
- 全局谱范数上界、采样 Jacobian 奇异值、固定点逆误差；
- $\eta_e,\eta_u$ 和二阶方向有限差分；
- 三个模型 seed 的完整结果，不只报告选中 seed。

## 失败解释规则

- 目标缺陷下降但收缩门失败：只能说明向量场更接近目标，不能称为动力学收缩；
- 收缩通过但在线误差退化：证书坐标可行，但观测注入 $B$ 不具备可用的重构性能；
- 收缩和在线均通过但 $\eta_e$ 或 $\eta_u$ 失败：结果来自近线性或状态无关变换，不能支持
  “真正非线性 $T_\phi$”的研究主张；
- 单 seed 通过而其余 seed 失败：记为优化不稳定，不解锁 test；
- 全部 validation 门通过：只称为 $n=31$ 声明状态域和有限时间范围内的数值证据。

## 实施阶段与停止条件

1. **数学单元测试**：核对 $A,F$ 分解、目标能量不等式、完整链式法则和观测器符号；
2. **结构单元测试**：零纤维、谱范数全局界、Jacobian 界、固定点逆和非线性审计；
3. **CPU 小样本检查**：固定 $B_0$ 验证目标与四项损失均有限；
4. **2060 1-epoch smoke**：检查显存、梯度、rollout 和 checkpoint；
5. **2060 正式三 seed 联合训练**：所有 seed 都必须运行完，科学门失败不停止其他 seed；
6. **validation-only 判定**：未通过则停止并报告，不自动追加训练或修改权重；
7. **locked test**：仅在 validation 总门通过后执行一次。

崩溃、非有限梯度、谱约束失效或输出不完整属于执行失败；冻结科学门未通过属于有效负结果。

## 计划产物

| 产物 | 计划位置 | 成功条件 |
|---|---|---|
| 实现入口 | tool/r5_nonlinear_target_conditional_residual_joint.py | 参数冻结且拒绝覆盖输出 |
| 网络与损失 | src/allen_cahn_certified_observer/ | 单元测试覆盖链式法则、全局界和四项损失 |
| 测试 | tests/ | 本地无 GPU 测试和 2060 完整测试通过 |
| checkpoint | out/\<new\>/checkpoints/ | 三个 seed 各一个，包含 $B,T_\phi$ 和冻结配置 |
| 正式结果 | out/\<new\>/results.json | validation、门、环境和精确 Git SHA 完整 |
| 正式日志 | out/\<new\>/run.log | 无 traceback、NaN 或静默中断 |
| 结果报告 | report/r5-nonlinear-target-conditional-residual-joint.md | 按冻结门逐项报告，不修改判据 |

正式入口命令为：

~~~bash
PYTHONPATH=src:tool python tool/r5_nonlinear_target_conditional_residual_joint.py \
  --nu 0.005 --grid-size 31 --sensor-count 3 \
  --seeds 1101 1102 1103 --epochs 80 \
  --instant-batch-size 512 --rollout-batch-size 4 \
  --rho 0.5 --device cuda \
  --checkpoint-dir out/<new>/checkpoints \
  --output out/<new>/results.json
~~~

## 当前状态

计划已按冻结配置执行。精确提交为
`32cc1ab3612e1a24e8a5c659aa734d9d90781d06`；三个模型 seed 均完成，但成功门为 0/3，
因此 locked test 未执行。正式结论见
`report/r5-nonlinear-target-conditional-residual-joint-20260824.md`。
