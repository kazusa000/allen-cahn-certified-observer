# R5：三传感器动力学联合训练实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-24
- Verification Status: UNVERIFIED — PLAN FROZEN BEFORE IMPLEMENTATION
- Version Label: code_plan_v1

## 研究问题

研究计划使用

\[
\partial_t u=Au+F(u),\qquad y=\mathcal C u,
\]

\[
\partial_t\hat u=A\hat u+F(\hat u)+B(y-\mathcal C\hat u),
\qquad e=\hat u-u,
\]

以及稳定目标

\[
z=T_\phi(e),\qquad \partial_tz=(A-\lambda I)z.
\]

由观测器的符号直接相减，本实验执行的精确误差方程为

\[
\partial_t e=Ae+F(u+e)-F(u)-B\mathcal C e.
\]

研究计划在线性段写出的 \((A+B\mathcal C)e\) 与该观测器符号不一致；代码以观测器方程
导出的负号为准，不通过改名掩盖这个差异。

前序确定性实验已经证明：固定总观测长度 0.20 时，三个传感器不能给出质量伴随注入的
全局半离散证书，但一般 (B) 与固定可逆 (T\) 在全部声明轨迹上通过直接收缩和在线门。
本实验不重复证明该结论，而回答两个更窄的问题：

1. 联合训练 (B\) 与 (T_\phi\) 能否同时保持三传感器的直接收缩，并显著减小相对固定稳定
   目标的动力学缺陷；
2. 该改善是否确实来自 (T_\phi\)，而不是只训练 (B\) 就能得到的结果。

## 冻结观测配置与模型

固定三个局部平均传感器为

\[
[0.1666667,0.2333333],\quad
[0.4666667,0.5333333],\quad
[0.7666667,0.8333333].
\]

总观测长度仍为 0.20。参数固定为

\[
\nu\in\{0.005,0.010,0.020\},\qquad
\lambda=0.1\nu\pi^2.
\]

每个 \(\nu\) 的 (B\) 与线性 (T\) 都从前序一般输出注入的 LMI 设计初始化。只训练围绕该
可行初始化的有界低模态残差：(B\) 的质量范数偏移不超过初始化的 25%，(T_\phi\) 的低模态
缩放范围固定为 \(\exp(\pm0.2231435513)=[0.8,1.25]\)。高频部分保持初始化，不让网络用
任意全维映射伪造动力学拟合。

Allen--Cahn 非线性增量取决于 (u\)，所以可训练实现使用状态条件化扩展

\[
z=T_\phi(u,e),
\]

并通过自动微分计算完整导数

\[
\partial_tz=D_uT_\phi(u,e)\,\partial_tu
          +D_eT_\phi(u,e)\,\partial_te.
\]

这只是研究计划中 (T_\phi(e)\) 在非线性阶段的实现扩展；报告必须同时保留原符号和扩展
原因，不能把有限样本结果写成全局算子恒等式。

## 联合损失

一步稳定损失严格沿用研究计划：

\[
\mathcal L_{\mathrm{stable}}
=\frac{\|T_\phi(u_{n+1},e_{n+1})-
S_{\Delta t}T_\phi(u_n,e_n)\|_h^2}
{\Delta t^2(\|e_n\|_h^2+\varepsilon)}.
\]

动力学缺陷直接检查同一个 (T_\phi\) 是否把实际误差动力学送到研究计划的稳定目标：

\[
\mathcal L_{\mathrm{defect}}
=\frac{\|\partial_tT_\phi(u,e)-(A-\lambda I)T_\phi(u,e)\|_h^2}
{\|e\|_h^2+\varepsilon}.
\]

直接收缩约束为

\[
\mathcal L_{\mathrm{contraction}}
=\left(
\max\left\{0,
\frac{\langle z,\partial_tz\rangle_h+\lambda\|z\|_h^2}
{\|z\|_h^2+\varepsilon}
\right\}\right)^2.
\]

另保留双向范数约束和 (B\) 相对 LMI 初始化的正则。主损失权重冻结为

\[
\mathcal L
=\mathcal L_{\mathrm{stable}}
+\mathcal L_{\mathrm{defect}}
+10\mathcal L_{\mathrm{contraction}}
+\mathcal L_{\mathrm{bi}}
+0.1\mathcal L_{B}.
\]

用户指出的 ODE 实验还使用 (TB=B\)。对状态条件化非线性映射，本实验只在一个预先声明
的对照配置中加入其微分对应物

\[
\mathcal L_{TB}
=\frac{\|D_eT_\phi(u,e)B-B\|_{h,F}^2}
{\|B\|_{h,F}^2+\varepsilon},
\]

权重为 1。它不是默认真理；只有验证结果优于不加该项的联合训练，才保留它。

## 消融矩阵

所有行使用相同训练/验证轨迹、初始化和评估代码：

| 名称 | (B\) | (T\) | 目的 |
|---|---|---|---|
| `fixed-B__identity-T` | 固定 LMI | 恒等 | 无变换基线 |
| `fixed-B__fixed-T` | 固定 LMI | 固定 LMI | 前序三传感器证书基线 |
| `train-B__identity-T` | 训练 | 恒等 | 判断只训练 (B\) 能做到多少 |
| `fixed-B__train-T` | 固定 LMI | 训练 | 判断只训练 (T_\phi\) 能做到多少 |
| `joint-native` | 训练 | 训练 | 主联合训练 |
| `joint-ode-direction` | 训练 | 训练 | 主损失再加 \(\mathcal L_{TB}\) |

前两行不训练，只做同一 validation 审计。后四行使用 seed 501、502、503；每个 seed 80
epoch，batch size 512，每 20 epoch 用当前 (B\) 刷新一次训练轨迹。训练集每个 \(\nu\)
取 16 条冻结 pilot，并加入第 4 模态和接近最小观测方向的双符号困难误差；validation 每个
\(\nu\) 取 8 条不相交 pilot 和同类困难误差。test 在选择期间保持锁定。

## 选择、成功与停止规则

### `n=31` validation-only 筛选

先分别在每个可训练配置内选择 seed，顺序固定为：

1. 全部 validation 样本满足
   \(-\langle z,\partial_tz\rangle_h/\|z\|_h^2\geq\lambda\)；
2. 最小收缩裕量更大；
3. 动力学缺陷 RMS 更小；
4. \(\nu=0.005\) 终点 \(\|e\|_h\) 中位数更小。

两个联合配置之间，`joint-ode-direction` 只有在全部结构、收缩和在线门都通过，且动力学缺陷
RMS 至少比 `joint-native` 低 10% 时才胜出；否则选择 `joint-native`。

### 三传感器动力学联合训练通过门

选中的联合配置必须同时满足：

1. 数值有限，(T_\phi(u,0)=0\)，声明样本上的 Jacobian 奇异值位于 `[0.25, 3.5]`；
2. validation 的最坏直接收缩率不低于对应 \(\lambda\)；
3. 每个 \(\nu\) 的终点误差中位数不超过 `fixed-B__fixed-T` 在线基线的 1.05 倍；
4. 动力学缺陷 RMS 至少比 `fixed-B__fixed-T` 和 `train-B__identity-T` 分别降低 25% 和 20%；
5. 动力学缺陷 95% 分位数至少比 `fixed-B__fixed-T` 降低 15%；
6. 相对 `fixed-B__train-T` 的动力学缺陷 RMS 至少降低 10%，用于证明联合训练而非只训练
   (T_\phi\) 的额外价值。

第 4 条是 (T_\phi\) 有价值的主要证据；第 6 条是 (B+T_\phi\) 联合训练有价值的主要证据。
任一条件失败，都必须按对应消融明确报告，不能把“损失下降”改写成“动力学问题已解决”。

### 扩展与 test 解锁

- 若 `n=31` 失败上述任一门，停止，不运行 test，也不扩展网格。
- 若 `n=31` 全部通过，只把选中的联合配置按相同超参数扩展到 `n=63` 和 `n=127`，仍使用
  三个冻结 seed；每个网格先过 validation 门，才解封该网格的 test 与幅值 0.01 共同正弦
  测量噪声 test。
- 四传感器固定质量伴随方案只引用前序全局证书作为正对照，不进行神经网络训练；两传感器
  失败路线只作为负对照，不重新消耗 GPU。

## 输出、复现与资源

- 实现入口：`tool/r5_three_sensor_dynamics_joint.py`。
- 正式测试：`tests/test_three_sensor_dynamics_joint.py` 加完整 `pytest`。
- 本机只运行静态检查、CPU 基线和最小 smoke；正式多 seed 训练放到 2060。
- 远端必须运行本分支已推送的精确提交，在独立 tmux 中执行；输出写入新的
  `out/<日期>-r5-three-sensor-dynamics-joint-2060-<提交>/`，不得覆盖旧输出。
- 主结果为 JSON，至少记录代码提交、设备、配置、seed、损失分布、收缩分布、在线误差、
  test 是否解锁和每条冻结门的布尔值。
- 正式命令为：

  ```bash
  PYTHONPATH=src python tool/r5_three_sensor_dynamics_joint.py \
    --grid-sizes 31 63 127 --seeds 501 502 503 \
    --epochs 80 --batch-size 512 --refresh-interval 20 \
    --device cuda --train-limit-per-nu 16 \
    --validation-limit-per-nu 8 --test-limit-per-nu 8 \
    --stress-truths-per-nu 2 \
    --checkpoint-dir out/<新目录>/checkpoints \
    --output out/<新目录>/results.json
  ```

- 远端进程退出码非零、JSON 缺失/为空或出现非有限数均为运行失败，不自动把部分输出当成
  结论。

## 预期解释边界

通过本实验最多能说明：在声明的三传感器、网格、参数、轨迹和误差范围内，联合学习的
(B+T_\phi\) 比单独训练其中一个更接近研究计划指定的稳定目标动力学，并保持直接收缩与
在线性能。它仍不是连续 Allen--Cahn PDE 上的全局算子恒等式或全局稳定定理。
