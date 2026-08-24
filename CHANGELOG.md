# 实验变更记录

## Unreleased

- 建立实验工程。
# Changelog

- Planned the R5 observation-injection repair: a five-sensor certified baseline, a two-sensor
  oblique modal injection feasibility gate, nonlinear/noise validation, and a training-only-after-
  feasibility compute policy.
- Added a general causal output-injection observer, unstable-mode observability diagnostics,
  pole-placement/Riccati/LMI modal designs, physical-gain and transient metrics, a five-sensor global
  semidiscrete margin audit, and the paired nonlinear/noise CPU experiment entry point.
- Added bounded low-mode joint training around the LMI output injection and its balanced invertible
  metric transform. The trainer uses direct contraction as the primary loss, target-dynamics defect as
  an auxiliary loss, structural invertibility bounds, on-policy refresh, fourth-mode and near-unobserved
  stress cases, and validation-only model selection before test/noise evaluation.
- Froze a three-configuration validation-only GPU screen so gain-only, balanced-joint, and flexible-
  joint residuals can be selected without test leakage before any multi-grid expansion.
- Regularized the zero-initialized gain-residual norm and added a pre-update finite-gradient gate after
  the first remote smoke exposed an undefined zero-norm derivative.

- Added the R5 direct transformed-error contraction audit, checkpoint replay, contraction-aware joint
  loss, finite-sample worst-margin diagnostics, and pre-registered GPU training screen.
- Completed the RTX 2060 direct-contraction screen: the selected weight-10 model improved the worst
  validation rate from -0.4810 to -0.4043 while preserving invertibility and online error, but did not
  obtain a positive finite-sample validation margin.

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
