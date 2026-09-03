# Allen-Cahn Direct Contraction Observer

High-accuracy Allen-Cahn solver and direct-contraction nonlinear observer correction under partial observations.

## 当前主线

本仓库只继续维护冻结的 `nu=0.005`、三传感器直接 fiber 收缩路线：

- `tool/r5_direct_fiber_multigrid_joint.py`：三网格联合训练与校准；
- `tool/r5_direct_fiber_adversarial_repair.py`：低模态困难点修复与 checkpoint 回放；
- `tool/r5_g_generalization_audit.py`：fresh validation 与 locked test；
- `tool/r5_h_state_ood_audit.py`：状态分布外泛化审计；
- `tool/r5_i_backstepping_nonlinear_remainder_audit.py`：冻结模型的预设目标缺陷与
  真实非线性余项诊断。

早期预设目标、局部证书、低频尾项和传感器数量路线已有独立归档实验，不再在本仓库重复
维护。2026-08-29 整理出的本地可恢复副本位于
`trash/2026-08-29-obsolete-r5-routes/`，清单见 `docs/cleanup-20260829.md`。

## 入口

- `design/`：实验方案与配置
- `src/`：正式实现
- `tests/`：正式测试
- `data/`：数据与来源说明
- `out/`：原始运行输出
- `report/`：整理后的分析与结论
- `tool/`：实验辅助工具
