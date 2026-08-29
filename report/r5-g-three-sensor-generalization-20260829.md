# R5-G：三传感器独立泛化审计

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-29
- Verification Status: VERIFIED — STRICT FRESH VALIDATION AND LOCKED TEST PASSED
- Version Label: exp_result_v1
- Code Commit: `fe785b5256dd752eb9d2e72077311f2acae60b8e`
- Checkpoint SHA-256: `83413559a98b9bb39226763ff3dd050610557fb6ad9b0037b7e23bb682f79d92`
- Validation Result SHA-256: `44876270036b1764c8a2d4c109dbe00740f6283785f4574141d1563ef35389fe`
- Locked Test Result SHA-256: `c690496cc4f800cfcd6ba998e594824f5f30516fa6c558942dd681649ecc6493`
- Node: RTX 2060 Mobile Linux

## 结论

冻结的 `nu=0.005, q=3` seed-1303 checkpoint 在没有任何优化步骤的前提下，通过了 fresh
validation seed 1871 和 locked test seed 1901。严格全点门和宽松实用门同时通过；checkpoint
在运行前后保持同一 SHA-256。

这将原来的“已消费校准集上可用”升级为：在声明的有限状态--误差分布、有限轨迹时段和本轮
半离散网格集合上，存在独立样本外收缩证据，并且同一共享模型能直接提升到未参与训练的网格。
它仍不是连续 PDE 紧集上的统一定理；其他扩散系数与传感器数量不在本实验范围内。

## Fresh validation

seed 1871；每个网格 4096 个 collocation。`n=47,95` 从未参与训练。

| 网格 | 最坏 collocation 余量 | 1% 分位余量 | 最坏轨迹余量 | 终点中位数/基线 | 终点最大值/基线 |
|---:|---:|---:|---:|---:|---:|
| 31 | 0.050471 | 0.259247 | 0.169918 | 0.8732 | 0.9702 |
| 47 | 0.023270 | 0.264614 | 0.236233 | 0.9551 | 1.0244 |
| 63 | 0.048289 | 0.275277 | 0.154971 | 0.8764 | 1.0066 |
| 95 | 0.057606 | 0.260579 | 0.221334 | 0.8192 | 0.9764 |
| 127 | 0.064795 | 0.267094 | 0.337812 | 0.9457 | 1.0129 |

所有 collocation 和轨迹通过率均为 100%。最低余量仍为严格正数，因此 test 按冻结合同解锁。

## Locked test

seed 1901；每个网格 8192 个 collocation。`n=191,255` 是未训练的网格外推。

| 网格 | 最坏 collocation 余量 | 1% 分位余量 | 最坏轨迹余量 | 终点中位数/基线 | 终点最大值/基线 |
|---:|---:|---:|---:|---:|---:|
| 31 | 0.029760 | 0.260809 | 0.232136 | 0.9956 | 1.0094 |
| 63 | 0.045949 | 0.261402 | 0.143225 | 0.8398 | 0.9936 |
| 127 | 0.042780 | 0.274104 | 0.291990 | 0.9728 | 0.9971 |
| 191 | 0.027359 | 0.273107 | 0.166726 | 0.9038 | 1.0140 |
| 255 | 0.025763 | 0.264685 | 0.182192 | 0.9185 | 1.0083 |

所有 test collocation 和轨迹通过率同样为 100%。最细网格没有出现系统性余量衰减，在线误差
中位数均不差于基线，最大值最多高约 1.4%。

## 结构和完整性

- 零纤维、残差谱界、Jacobian 界和固定点数值求逆全部通过；
- validation 与 test 使用相同 checkpoint 哈希；
- evaluation-only 路径没有创建优化器，也没有写回 checkpoint；
- 远程工作树在运行前后均干净，HEAD 与已推送 commit 完全一致；
- 完整测试：111 passed；两个正式进程退出码均为 0。

## 证据边界

本轮只覆盖 `nu=0.005`、固定三传感器几何、当前低模态有限域和 `t in [0,1]`。未见网格通过
证明了离散提升能力，但有限的五张 test 网格不能替代网格一致解析估计。更高频、多界面、长
轨迹、噪声和传感器偏移没有纳入本轮结论。

## 主实验原始输出

- validation：`/home/wjj/work/main/phd/project/project1/experiment/03-allen-cahn-direct-contraction-observer/out/2026-08-29-15-11-r5-g-validation-fe785b5/`
- locked test：`/home/wjj/work/main/phd/project/project1/experiment/03-allen-cahn-direct-contraction-observer/out/2026-08-29-15-14-r5-g-locked-test-fe785b5/`
