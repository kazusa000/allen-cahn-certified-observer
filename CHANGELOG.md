# 实验变更记录

## Unreleased

- 建立实验工程。
# Changelog

## Unreleased

- Added the R5-A Allen–Cahn reference model, discrete energy diagnostic, exact fixed-width local-average observations, and causal constant-gain nudging baseline.
- Added the R5-B local incremental Jacobian/remainder diagnostics and offline causal observer rollout harness.
- Added the R5-C offline certificate interface and fiber/direction/local-invertibility audit.
- Added the R5 pilot formal contract for state/error domains, grids, splits, noise, and compute gates.
- Corrected causal nudging to use the physical mass-adjoint observation injection for cross-grid comparability.
- Added deterministic R5 pilot case generation and an exploratory local baseline sweep tool.
- Added the CPU-only causal state-conditioned linear residual correction fit for R5-D, with a fixed physical-gain safeguard.
- Added the reproducible held-out R5-D smoke runner with physical-error, energy-defect, and measurement-noise diagnostics.
- Added the optional PyTorch R5-E joint correction/certificate GPU pilot runner.
