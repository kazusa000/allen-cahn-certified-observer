# 实验变更记录

## Unreleased

- 建立实验工程。
# Changelog

## Unreleased

- Added the R5 low-frequency certificate mode with grid-consistent physical sine projections,
  projected online correction, and separate high-frequency tail/coupling audits.
- Added the R5-A Allen–Cahn reference model, discrete energy diagnostic, exact fixed-width local-average observations, and causal constant-gain nudging baseline.
- Added the R5-B local incremental Jacobian/remainder diagnostics and offline causal observer rollout harness.
- Added the R5-C offline certificate interface and fiber/direction/local-invertibility audit.
- Added the R5 pilot formal contract for state/error domains, grids, splits, noise, and compute gates.
- Corrected causal nudging to use the physical mass-adjoint observation injection for cross-grid comparability.
- Added deterministic R5 pilot case generation and an exploratory local baseline sweep tool.
- Added the CPU-only causal state-conditioned linear residual correction fit for R5-D, with a fixed physical-gain safeguard.
- Added the reproducible held-out R5-D smoke runner with physical-error, energy-defect, and measurement-noise diagnostics.
- Added the optional PyTorch R5-E joint correction/certificate GPU pilot runner.
- Added independent R5-F checkpoint replay, baseline comparison, and measurement-noise verification.
- Added the R5 ablation matrix runner for fixed-gain, state-conditioning, certificate, direction,
  and bounded-gate factors.
- Recorded the completed 2060 R5 ablation sweep and its certificate/online-coupling conclusion.
- Added the T--K-style R5 joint trainer that optimizes the discrete stable-target loss through both
  the state-conditioned certificate and the deployable gain network.
- Added the R5 joint dynamics-defect loss, two-sided invertibility loss, and structurally bounded
  state-conditioned nullspace scaling; completed the formal 2060 multi-grid run.
- Corrected the R5 stable target to the research-plan diffusion operator, normalized the discrete
  stable loss, added current-observer trajectory refresh, and added validation-rollout model selection.
- Completed the 2060 screening and formal multi-grid replay for the corrected target; the selected
  observer improved noiseless held-out error on all three grids while the dynamics defect remained open.
- Added the pre-registered R5 dynamics-defect repair screen with mixed trajectory replay, structurally
  invertible Givens nullspace mixing, staged-to-joint training, checkpoint persistence, and split-wise
  defect-distribution audits.
- Added the second-stage T--K structure screen with a bounded triangular observed-to-nullspace shear,
  configurable stable-loss weight, and gain-range ablations after the first repair screen failed its
  pre-registered defect-reduction gate.
- Added separate gain/certificate learning rates, staged certificate-first training, gradient clipping,
  and bounded online-rollout failure handling for the triangular structure screen.
- Added a state-wise trust region and normalized deviation penalty around the initialized correction
  operator to keep the joint observer rollout in its numerically stable neighborhood.
- Added physical mass-adjoint correction variants and selected jointly learned constant positive sensor
  gains for the formal screen, matching the correction-operator form declared in the research plan.
- Removed the triangular transform's non-smooth zero-norm second derivative and added immediate
  non-finite loss/gradient checks after it was found to contaminate preliminary online rollouts.
- Aligned the stage-two constant correction initialization with the frozen gain-0.10 validation
  comparator after the first structure screen exposed an incompatible gain-0.02 trust region.
- Completed the calibrated R5 dynamics-defect repair screen on the RTX 2060; validation defect RMS
  improved by 10.5% but failed the frozen 50% progress gate, so no multi-grid expansion was run.
