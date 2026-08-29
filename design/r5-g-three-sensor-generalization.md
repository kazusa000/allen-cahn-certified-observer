# R5-G：三传感器泛化审计

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + run
- Origin Date: 2026-08-29
- Verification Status: COMPLETED — STRICT VALIDATION AND TEST PASSED
- Version Label: code_plan_v2

## 研究问题

本实验只回答一个问题：已冻结的 `nu=0.005, q=3` checkpoint 能否通过独立样本与未见网格
审计。不同扩散系数和不同传感器数量不属于本实验范围。

## 冻结对象

- 方程参数：`nu=0.005`；
- 传感器：三个冻结的局部平均区间；
- checkpoint：seed 1303、`rho=0.9`、50% 增益信赖域；
- evaluation-only：不创建优化器，不执行梯度更新，不写回 checkpoint；
- validation seed：1871；locked test seed：1901；
- validation 网格：31、47、63、95、127，每网格 4096 个 collocation；
- locked test 网格：31、63、127、191、255，每网格 8192 个 collocation；
- 时间区间：`[0,1]`。

validation 的实用门通过后才允许读取 locked test。严格门同时报告，但不是唯一解锁条件。

## 判据

严格门要求所有 collocation 与轨迹余量非负，在线终点误差中位数不超过基线 1.05 倍、最大值
不超过 1.10 倍，并通过全部结构检查。

实用门允许有限尾部误差，但每个网格仍须同时满足：

- collocation 通过率至少 99.5%，1% 分位余量非负，最坏余量不低于 -0.10；
- 轨迹通过率至少 99%，1% 分位余量非负，最坏余量不低于 -0.02；
- 在线终点误差中位数不超过基线 1.10 倍，最大值不超过 1.20 倍；
- 零纤维、谱界、Jacobian 和数值求逆结构门全部通过。

## 完成状态

fresh validation 和 locked test 均通过严格门与实用门。所有采样余量为正，checkpoint 哈希在
审计前后保持不变。完整结果见 `report/r5-g-three-sensor-generalization-20260829.md`。
