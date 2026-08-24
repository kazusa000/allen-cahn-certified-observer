# R5：真实误差动力学上的跨网格低频条件变换联合训练计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-24
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

## 研究问题

固定

\[
\nu=0.005,
\qquad q=3,
\qquad n\in\{31,63,127\}.
\]

研究计划中的系统和观测器为

\[
\partial_tu=Au+F(u),
\qquad y=\mathcal C u,
\]

\[
\partial_t\hat u=A\hat u+F(\hat u)+B(y-\mathcal C\hat u),
\qquad e=\hat u-u,
\]

其中

\[
A=\nu\Delta_h,
\qquad F(v)=v-v^3.
\]

因此真实误差动力学是

\[
\partial_te=Ae+F(u+e)-F(u)-B\mathcal C e.
\]

本实验检验：同一个跨网格低模态增益和同一个条件可逆低模态变换，能否在三个离散网格上
同时使真实误差动力学满足有限样本收缩，并且保持在线观测误差不劣于固定 LMI 基线。

本实验不再要求变换后的动力学等于另一个人为指定的 Allen--Cahn 方程。旧目标

\[
Az+F(u+z)-F(u)-(1+\lambda)z
\]

只保留为历史对照，不进入训练损失或成功门。原因是它在线性化处要求两个闭环矩阵相似，
而现有三传感器增益不满足该必要条件；同时，$F$ 的逐点形式在一般坐标变换下并不保持不变。

## 跨网格变换

令 $V_{4,h}$ 是前四个 $M_h$ 正交归一的 Dirichlet 模态，$V_{8,h}$ 是前八个模态，

\[
a=V_{8,h}^{\mathsf T}M_hu,
\qquad b=V_{4,h}^{\mathsf T}M_he,
\qquad \Pi_{4,h}=V_{4,h}V_{4,h}^{\mathsf T}M_h.
\]

只在四维不稳定子空间学习条件变换，高频误差保持原坐标：

\[
T_{\phi,h}(u,e)
=V_{4,h}T_0\left[b+g_\phi(a,b)-g_\phi(a,0)\right]
+(I-\Pi_{4,h})e.
\]

$T_0\in\mathbb R^{4\times4}$ 是 $n=31$ 三传感器 LMI 度量的平衡平方根。$g_\phi$ 是带
$a$ 加性条件输入的多层 $\tanh$ 网络。三个网格共享同一个 $T_0$ 和同一组 $\phi$，网络参数
维数不随 $n$ 增长。

对误差通道各线性层执行硬谱投影，使

\[
\sup_{a,b}\|D_bg_\phi(a,b)\|_2\leq\rho<1.
\]

注意这里没有额外的系数 $2$：$g_\phi(a,0)$ 对 $b$ 的导数为零。因此

\[
1-\rho
\leq\sigma_{\min}\!\left(I+D_bg_\phi\right)
\leq\sigma_{\max}\!\left(I+D_bg_\phi\right)
\leq1+\rho.
\]

低频映射关于 $b$ 全局可逆，高频部分是恒等映射，所以 $T_{\phi,h}(u,\cdot)$ 在每个网格上
全局可逆，并自动满足 $T_{\phi,h}(u,0)=0$。

## 跨网格增益

令

\[
B_h=V_{4,h}\Beta,
\qquad \Beta\in\mathbb R^{4\times3}.
\]

$\Beta$ 使用 $n=31$ LMI 增益转换到物理模态坐标后的值初始化，并与 $\phi$ 从第一个 epoch
起联合训练。三个网格共享同一个 $\Beta$。训练期间执行硬信赖域投影

\[
\|\Beta-\Beta_0\|_F\leq0.25\|\Beta_0\|_F.
\]

四传感器质量伴随增益只作为正对照，用于确认代码和数据域能够识别已知的全局半离散收缩
情形，不与三传感器主模型混合训练。

## 主判据与损失

令

\[
z=T_{\phi,h}(u,e).
\]

完全使用链式法则计算

\[
\partial_tz
=D_uT_{\phi,h}(u,e)\partial_tu
+D_eT_{\phi,h}(u,e)\partial_te.
\]

定义真实动力学上的样本收缩率

\[
r_{\phi,B,h}(u,e)
=-\frac{\langle z,\partial_tz\rangle_h}
{\|z\|_h^2+\varepsilon}.
\]

要求率固定为研究计划沿用的

\[
\alpha=0.1\nu\pi^2.
\]

训练目标为

\[
\mathcal L
=\mu\,\operatorname{CVaR}_{10\%}
\left([\alpha-r_{\phi,B,h}(u,e)]_+^2\right)
+\mathcal L_{\mathrm{在线误差}}
+\gamma\frac{\|\Beta-\Beta_0\|_F^2}{\|\Beta_0\|_F^2}.
\]

其中 $\mu$ 采用乘子式递增，使最坏样本约束优先于平均在线性能。可逆性由网络结构硬保证，
数值逆只作诊断，不再把“逆误差”当作可以与动力学折中的软损失。也不设置人为的
“非线性程度必须超过某个阈值”；$T_\phi$ 是否有价值由它相对“只训练 $B$、固定 $T_0$”
消融的收缩域、收缩余量和在线误差改善决定。

严格表述上，本轮证明对象是零误差纤维的有限样本 Lyapunov 收缩证据，不把它表述成任意
两条估计轨迹之间的全局微分收缩定理。

## 数据域和无泄漏划分

每个网格独立生成但按同一物理规则缩放的样本：

- $u$ 使用前八个模态的随机组合，并缩放到 $\|u\|_\infty\leq1.25$；
- $e$ 覆盖前四模态、第五至第十二模态及其混合方向；
- $0.02\leq\|e\|_h\leq0.8$，半径使用对数均匀采样；
- 显式加入前十二个单模态正负方向、第四不稳定模态和三传感器最小观测方向；
- train、validation、locked test 使用互不重叠的确定性 seeds；
- validation 和 locked test 的随机样本量均不少于每网格 2048 个。

在线轨迹使用独立初值 seeds。训练在线误差在 $n=31$ 上计算；三个网格的正式 validation 和
locked test 都使用高精度因果观测器重新积分并报告终点与全时域误差。这样既保持 GPU 训练
可承受，又不把跨网格结论建立在训练用显式时间步上。

## 对照与消融

冻结以下比较：

1. 固定 $B_0$、固定 $T_0$；
2. 联合训练 $B,T_\phi$；
3. 使用联合模型学得的 $B$，但把 $T_\phi$ 退回固定 $T_0$；
4. 四传感器质量伴随正对照。

比较 2 和 3 回答 $T_\phi$ 的增量价值；比较 1 和 3 回答联合训练中 $B$ 的价值。不得把二者
混为一个归因结论。

## Validation 门与停止规则

每个模型 seed 必须同时满足：

1. 三个网格上的数值均有限；
2. 零纤维、硬谱界、固定点逆和解析 Jacobian 全局界通过；
3. 每个网格全部 validation collocation 满足
   $r_{\phi,B,h}(u,e)\geq\alpha$；
4. 每个网格的 validation 因果轨迹保存点也全部满足同一收缩门；
5. 每个网格在线终点误差中位数不超过固定 $B_0$ 基线的 $1.05$ 倍，最大值不超过
   $1.10$ 倍；
6. 至少 $2/3$ 个模型 seed 通过第 1--5 项。

只有第 6 项通过才对 locked test 计算一次。若失败，test 保持未读，并按预注册报告最先失败
的门，不修改阈值、不自动重训、不把有限样本结果写成连续 PDE 证明。

## 正式配置与产物

- 模型 seeds：1301、1302、1303；
- epoch：80；每 epoch 24 个即时动力学 batch；
- 即时 batch：256；$\rho=0.35$；隐藏层 3、宽度 64；
- $\Beta$ 学习率 $3\times10^{-4}$，$\phi$ 学习率 $10^{-3}$；
- RTX 2060 正式运行，硬超时 12 小时，30 秒检查进程和日志增长；
- 入口：`tool/r5_direct_fiber_multigrid_joint.py`；
- 正式实现：`src/allen_cahn_certified_observer/direct_fiber.py`；
- 正式测试：`tests/test_direct_fiber.py`；
- 原始结果：新的 `out/<时间>-r5-direct-fiber-multigrid-joint/`；
- 整理结论：`report/r5-direct-fiber-multigrid-joint-20260824.md`。

## 预注册修订：正式运行前的容量筛选

首个单 seed、80 epoch pilot 后发现：LMI 使用 `eigh` 模态 $V_{\mathrm{eigh}}$，跨网格网络使用
固定正弦模态 $V_{\sin}$，两者满足

\[
V_{\mathrm{eigh}}=V_{\sin}R,
\qquad R=\operatorname{diag}(-1,-1,1,1).
\]

旧实现搬运 $B_0,T_0$ 时遗漏了 $R$，所以旧 pilot 的固定基线并不是原 LMI 系统；此前所有
CVaR 平台和容量筛选数字均作废，C2/C3 运行已停止，locked test 从未读取。

修正后的固定坐标初始化必须为

\[
\Beta_0=\sqrt h\,RK,
\qquad
T_0=RP^{1/2}R^{\mathsf T}.
\]

正式运行前新增硬回归门：固定正弦坐标提升后的 $B_0$ 必须逐项等于 LMI 的网格增益，且由
$T_0^{\mathsf T}T_0$ 重新计算的收缩率必须在 $10^{-10}$ 内复现原 LMI 收缩率。只有该门通过
后才重新运行原始 $\rho=0.35$、25% 增益域基线，随后才能判断是否需要以下容量筛选。

若修正基线仍显示两个硬预算饱和且最坏余量为负，则在保持同一 calibration split、同一模型
seed 1301、同一训练步数和 test 锁定的条件下，只筛选
以下三组结构容量：

| 方案 | $\rho$ | 增益信赖域 | 误差输入尺度 |
|---|---:|---:|---:|
| C1 | 0.65 | 0.50 | 1 |
| C2 | 0.65 | 0.50 | 4 |
| C3 | 0.85 | 0.75 | 4 |

误差尺度化定义为

\[
g_{\phi,s}(a,b)=s^{-1}\bar g_\phi(sa,sb),\qquad s\geq1,
\]

并在每层使用等价的加性条件缩放。于是 $D_bg_{\phi,s}$ 中输入的 $s$ 与输出的 $s^{-1}$
抵消，原有 $\|D_bg_{\phi,s}\|_2\leq\rho<1$ 和全局可逆性不变，但实际误差域不再被限制
在 $\tanh$ 的近线性区。

容量方案按以下顺序选择：先要求结构门和三个网格在线不退化门全部通过，再最大化三个网格中
最坏的 collocation 余量，再最大化最坏轨迹余量。筛选只用于选结构，不产生正式结论。

选定结构后，正式三 seed 运行把 validation collocation/trajectory seed 从已使用的 1801 改为
全新的 1851；locked test seed 1901 不变且此前未生成。如果 C1--C3 都没有使最坏余量继续
显著朝零移动，则停止三 seed 重复并报告当前低四模态条件残差模型的容量不足，不事后缩小
状态/误差域。
