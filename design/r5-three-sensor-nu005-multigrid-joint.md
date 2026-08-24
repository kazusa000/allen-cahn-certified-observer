# R5：三传感器、nu=0.005 的多网格联合训练计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-24
- Verification Status: UNVERIFIED — FROZEN BEFORE IMPLEMENTATION
- Version Label: code_plan_v1

## Experiment Overview

- **Title**：三传感器欠通道条件下 \(B+T_\phi\) 联合训练的网格趋势
- **Objective**：只在 \(\nu=0.005\) 下，比较 `n=31,63,127` 独立联合训练后的动力学缺陷、
  直接收缩、在线误差和可逆性，判断现有结果是否随网格加密保持。
- **Hypothesis**：三个传感器面对 4 个不稳定模态时，联合训练的 \(T_\phi\) 在三个网格上都
  能相对固定 LMI \(B,T\) 降低至少 25% 的 validation 动力学缺陷，同时保持全部困难轨迹的
  指定收缩率。
- **Type**：validation-only GPU training + multigrid trend analysis

## 为什么只取 nu=0.005

固定三个传感器时，`n=31` 的不稳定模态数为

| \(\nu\) | 不稳定模态数 | 传感器数 |
|---:|---:|---:|
| 0.005 | 4 | 3 |
| 0.010 | 3 | 3 |
| 0.020 | 2 | 3 |

所以只有 \(\nu=0.005\) 直接对应博士课题要检验的“观测通道少于不稳定模态，但借助一般
\(B\) 与变换 \(T_\phi\) 仍得到稳定重构”。上一轮同时训练三个 \(\nu\) 是跨参数鲁棒性实验，
不能继续作为 \(T_\phi\) 必要性的主门。本轮不使用 \(\nu=0.010,0.020\)，也不让它们参与
模型选择或趋势判定。

上一轮多参数模型在 \(\nu=0.005,n=31\) 上的分层结果只用于提出本轮假设，不作为本轮正式
证据，不参与 seed 选择。

## 冻结数学对象

沿用研究计划

\[
\partial_t u=Au+F(u),\qquad y=\mathcal C u,
\]

\[
\partial_t\hat u=A\hat u+F(\hat u)+B(y-\mathcal C\hat u),
\]

以及由该观测器直接导出的

\[
\partial_t e=Ae+F(u+e)-F(u)-B\mathcal C e.
\]

固定

\[
\nu=0.005,\qquad \lambda=0.1\nu\pi^2,
\]

和三个局部平均传感器

\[
[0.1666667,0.2333333],\quad
[0.4666667,0.5333333],\quad
[0.7666667,0.8333333].
\]

总观测长度为 0.20。每个网格分别计算自己的 LMI \(B,T\) 初始化；网格之间不共享参数，
只共享模型结构、训练数据生成规则、seed、损失权重和验收规则。

只训练原生联合模型

\[
B+T_\phi(u,e),
\]

不训练 \(T=I\)、只训练 \(B\)、只训练 \(T_\phi\) 或 ODE 方向约束变体。固定 LMI \(B,T\)
只做不训练的同网格参照。

## 冻结损失

使用上一轮已经实现并通过测试的同一组损失：

\[
\mathcal L
=\mathcal L_{\mathrm{stable}}
+\mathcal L_{\mathrm{defect}}
+10\mathcal L_{\mathrm{contraction}}
+\mathcal L_{\mathrm{bi}}
+0.1\mathcal L_B.
\]

其中稳定目标严格保持为

\[
\partial_tz=(A-\lambda I)z,
\qquad z=T_\phi(u,e),
\]

不加入 \(D_eT_\phi B\approx B\)，不调整权重，不因某个网格的中间结果改变 epoch 或学习率。

## 训练设置

- 网格：`n in {31, 63, 127}`，全部必须运行，任何一个网格失败科学门都不能阻断其余网格。
- seed：501、502、503。
- 每个 seed：80 epoch，batch size 512，每 20 epoch 刷新当前 \(B\) 的训练轨迹。
- 学习率：\(B\) 为 `5e-4`，\(T_\phi\) 为 `1e-3`。
- \(B\) 信赖域：相对 LMI 初始化 0.25。
- \(T_\phi\) 低模态缩放：`exp(+-0.2231435513)=[0.8,1.25]`。
- train：16 条冻结 pilot，加第 4 模态与接近最小观测方向的双符号困难误差。
- validation：8 条不相交 pilot，加相同类型困难误差。
- 不运行 test；本轮只回答 validation 网格趋势，避免把 test 参与路线开发。

每个网格内按以下顺序选择 seed：

1. validation 全部采样点满足
   \(-\langle z,\partial_tz\rangle_h/\|z\|_h^2\geq\lambda\)；
2. 最坏收缩裕量更大；
3. 动力学缺陷 RMS 更小；
4. validation 终点 \(\|e\|_h\) 中位数更小。

## 单网格门

每个 `n` 独立报告并判定：

1. 数值有限，\(T_\phi(u,0)=0\)，Jacobian 奇异值位于 `[0.25,3.5]`；
2. validation 全部采样点的直接收缩率不低于 \(\lambda\)；
3. validation 动力学缺陷 RMS 相对同网格固定 LMI \(B,T\) 至少降低 25%；
4. validation 动力学缺陷 95% 分位相对固定 LMI \(B,T\) 至少降低 15%；
5. validation 终点误差中位数不超过同网格固定 LMI \(B\) 的 1.05 倍。

单网格失败不停止训练，只在最终趋势中标记。

## 趋势判定

令

\[
r_n=
\frac{\text{联合训练在网格 }n\text{ 的动力学缺陷 RMS}}
{\text{固定 LMI }B,T\text{ 在网格 }n\text{ 的动力学缺陷 RMS}}.
\]

报告 \(r_{31},r_{63},r_{127}\)、三个最坏收缩裕量、三个在线误差比例和 Jacobian 奇异值范围。

只有同时满足以下条件，才称为“当前离散范围内网格稳健”：

1. 三个网格的全部单网格门通过；
2. \(\max r_n-\min r_n\leq0.10\)；
3. 动力学缺陷 RMS、最坏收缩裕量和在线误差比例没有随网格加密出现单调恶化。

第 3 条只做描述性趋势判定，不拟合只有三个点的统计模型，也不外推到连续 PDE 极限。即使
三网格通过，也只能称为半离散数值证据，不能写成连续 PDE 定理。

## Setup

- **Language/Framework**：Python 3.12、PyTorch 2.13、NumPy、SciPy、CVXPY。
- **Working Directory**：当前 `three-sensor-dynamics-joint` experiment worktree。
- **Environment**：2060，RTX 2060 6 GB，CUDA。
- **Timeout**：12 小时硬超时；每 30--60 秒监控进程、日志和 GPU。
- **Expected Output**：不可覆盖的 JSON、每网格三个 checkpoint、运行日志和退出状态。
- **Entry Command**：

  ```bash
  PYTHONPATH=src:tool python tool/r5_three_sensor_nu005_multigrid_joint.py \
    --seeds 501 502 503 --epochs 80 --batch-size 512 \
    --refresh-interval 20 --device cuda \
    --train-limit-per-nu 16 --validation-limit-per-nu 8 \
    --stress-truths-per-nu 2 \
    --checkpoint-dir out/<新目录>/checkpoints \
    --output out/<新目录>/results.json
  ```

- 正式运行前提交、推送并核验本地、跟踪分支和远程 SHA 完全一致。

## Analysis Plan

- **Primary metric**：三个网格各自的 validation 动力学缺陷 RMS 相对固定 LMI \(B,T\) 的比例。
- **Safety metric**：最坏直接收缩裕量。
- **Secondary metrics**：缺陷 95% 分位、收缩通过率、在线终点误差比例、Jacobian 奇异值。
- **Success criterion**：三网格全部通过单网格门和上述趋势判定。
- **Stopping rule**：执行崩溃与科学门失败分开报告；不自动重试崩溃，不用部分网格输出冒充
  完整趋势。
