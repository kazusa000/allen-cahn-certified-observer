# R5：非线性目标缺陷与收缩功率的四模态频带审计

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-24
- Verification Status: VERIFIED — EXECUTED; LOW-FREQUENCY-DOMINANT FAILURE
- Version Label: executed_audit_v1

## 问题

上一轮使用

\[
G_{\mathrm{all}}(u,z)
=Az+F(u+z)-F(u)-(1+\lambda)z.
\]

其中 $-(1+\lambda)z$ 对全部 31 个离散模态施加同样的额外项。该设计可能重新引入此前
全频统一谱移的问题，但总缺陷和总收缩失败本身不能确定失败来自低频还是高频。

本审计只回答：正式三个 checkpoint 的目标缺陷与实际收缩功率主要来自 $\Pi_4$ 低频还是
$I-\Pi_4$ 高频，以及改用

\[
G_{\mathrm{low}}(u,z)
=Az+F(u+z)-F(u)-(1+\lambda)\Pi_4z
\]

是否值得进入下一轮训练。当前目标、模型、checkpoint 和 validation 数据均不修改；locked
test 不读取。

## 投影

令 $V_4$ 为 $n=31$ 离散 Dirichlet Laplacian 的前四个物理正弦模态组成的欧氏正交基。由于
$M_h=hI$，定义

\[
\Pi_4=V_4V_4^\top,
\qquad Q_4=I-\Pi_4
\]

同时也是 $M_h$ 正交投影。对任意 $v$，写成

\[
v_L=\Pi_4v,\qquad v_H=Q_4v.
\]

## 目标缺陷分解

沿用正式训练中的完整链式法则，定义

\[
r_{\mathrm{all}}
=\partial_tz_{\mathrm{act}}-G_{\mathrm{all}}(u,z).
\]

审计

\[
r_L=\Pi_4r_{\mathrm{all}},
\qquad r_H=Q_4r_{\mathrm{all}}.
\]

低频、高频缺陷使用与正式实验相同的总误差归一化：

\[
d_L^2=\frac{\|r_L\|_h^2}{\|e\|_h^2+\varepsilon},
\qquad
d_H^2=\frac{\|r_H\|_h^2}{\|e\|_h^2+\varepsilon}.
\]

由于投影正交，$d_L^2+d_H^2=d_{\mathrm{all}}^2$，舍入误差除外。主要归因量为池化损失
占比

\[
s_{r,H}=\frac{\sum d_H^2}{\sum(d_L^2+d_H^2)},
\qquad s_{r,L}=1-s_{r,H}.
\]

同时报告 RMS、95% 分位、最大值和每条 validation 轨迹的分解。

不重新训练即可计算低频阻尼目标的反事实缺陷。因为

\[
G_{\mathrm{low}}-G_{\mathrm{all}}=(1+\lambda)z_H,
\]

所以

\[
r_{\mathrm{low}}
=r_{\mathrm{all}}-(1+\lambda)z_H.
\]

报告 $r_{\mathrm{low}}$ 相对 $r_{\mathrm{all}}$ 的 RMS 和 95% 分位变化。该量只说明当前
checkpoint 对新目标的瞬时匹配程度，不等同于重新训练结果。

## 收缩功率分解

实际变换动力学的收缩功率定义为

\[
P=-\langle z,\partial_tz_{\mathrm{act}}\rangle_h.
\]

利用正交投影分解为

\[
P_L=-\langle z_L,(\partial_tz_{\mathrm{act}})_L\rangle_h,
\qquad
P_H=-\langle z_H,(\partial_tz_{\mathrm{act}})_H\rangle_h,
\]

并有 $P=P_L+P_H$。相对要求收缩率 $\lambda$ 的频带功率裕量为

\[
M_L=P_L-\lambda\|z_L\|_h^2,
\qquad
M_H=P_H-\lambda\|z_H\|_h^2,
\qquad M=M_L+M_H.
\]

在 $M<0$ 的失败样本上，定义负裕量负担

\[
b_L=[-M_L]_+,qquad b_H=[-M_H]_+,
\]

并报告池化高频占比

\[
s_{M,H}=\frac{\sum b_H}{\sum(b_L+b_H)}.
\]

同时报告低频、高频各自的功率、归一化收缩率、最坏裕量、失败率，以及总失败样本中
“仅低频失败”“仅高频失败”“两者都失败”的数量。

## 对照、数据和一致性检查

- checkpoint：seed 1101、1102、1103；
- 数据：只重建原正式 validation 16 条轨迹、1616 个保存点；
- 对照：固定 $B_0,T_0$；
- 不读取 train 结果以替代 validation，不读取 locked test；
- 审计重新计算的总缺陷 RMS、95% 分位、最坏总收缩裕量和收缩通过率必须与正式
  `results.json` 在数值容差内一致；
- 检查 $r_L+r_H=r$、$P_L+P_H=P$ 和 $M_L+M_H=M$ 的最大加和误差。

## 预注册归因与决策规则

对三个 learned checkpoint 分别计算 $s_{r,H}$ 和 $s_{M,H}$：

1. 至少 2/3 checkpoint 的占比和三个 checkpoint 的池化占比都不小于 0.60，才称对应失败
   **主要来自高频**；
2. 至少 2/3 checkpoint 的占比和池化占比都不大于 0.40，才称对应失败**主要来自低频**；
3. 其余情况称为混合频带，不能把失败单独归因于某一部分。

只有以下条件同时成立，才建议把 $-(1+\lambda)\Pi_4z$ 作为下一轮的主要候选：

1. 离散第五模态满足
   \[
   \nu\mu_{5,h}-1\geq\lambda,
   \]
   从而投影目标仍至少以 $\lambda$ 收缩；
2. 三个 checkpoint 的反事实缺陷 RMS 和 95% 分位均至少改善 10%；
3. 目标缺陷主要来自高频，且实际收缩功率失败不被判为主要来自低频。

若第二项不成立，则统一高频阻尼不是当前目标缺陷的主要矛盾。若收缩功率主要在低频失败，
则改变高频目标项不能直接解释或修复当前收缩失败，应优先保留 $T_0$ 的低频稳定几何。

## 计划产物

| 产物 | 位置 |
|---|---|
| 审计入口 | `tool/r5_nonlinear_target_frequency_audit.py` |
| 单元测试 | `tests/test_nonlinear_target_frequency_audit.py` |
| 原始结果 | `out/<new>/frequency-audit.json` |
| 结论报告 | `report/r5-nonlinear-target-frequency-audit-20260824.md` |

## 当前状态

审计已在精确提交 `6cfef8d389294b5c0e43853a9b9bfd715e9fdc1c` 上完成。三个 checkpoint
的目标缺陷高频池化占比为 5.11%，收缩失败高频负裕量负担池化占比为 7.89%，两者均判为
低频主导。投影阻尼反事实未达到 10% 改善门，因此不建议把
$-(1+\lambda)\Pi_4z$ 作为下一轮主要修复。完整结论见
`report/r5-nonlinear-target-frequency-audit-20260824.md`。
