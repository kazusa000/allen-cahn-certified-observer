# 实验变更记录

## Unreleased

- Froze a new evaluation-only state-distribution OOD audit for the existing
  `nu=0.005`, three-sensor checkpoint. The prior IID validation and locked-test
  conclusion remains unchanged; the new audit separately covers frequency,
  localization, interface complexity, and initial-error magnitude.

- Froze the R5-G three-sensor generalization audit at `nu=0.005`. The checkpoint is evaluated without
  optimization on fresh seeds and unseen grids. Added separate strict-certificate metrics and a deliberately
  tolerant practical gate so isolated tail points do not alone block empirical evaluation.
- Implemented the evaluation-only G1 runner for arbitrary unseen grids, immutable checkpoint hashing,
  validation-gated locked-test access, refreshed final-checkpoint ablations, and reusable strict/practical
  decision rules with regression tests for the frozen three-sensor protocol.
- Completed G1 on the RTX 2060. Fresh validation and locked test both passed the strict and practical
  gates on every declared grid, including unseen grids 47, 95, 191, and 255; all sampled contraction
  margins remained positive and the frozen checkpoint hash was unchanged.
- Renamed the Project 1 experiment entry to `03-allen-cahn-direct-contraction-observer` so the
  repository name matches the evidence-supported direct-contraction route rather than implying a
  completed global certificate.
- Reorganized reports so the active directory contains only the direct-fiber route; moved shared
  pre-split R5 reports to `archive/report/r5-legacy/` without changing their content.
- Completed the practical adversarial-repair target: the selected joint
  `B+T_phi` checkpoint passed collocation, trajectory, online, and structural
  gates on grids 31, 63, and 127, with worst margins `+0.12595`, `+0.02156`,
  and `+0.07083`. Stopped the stricter third multi-seed continuation after the
  user chose not to optimize around isolated 1-in-4096 tail points; fresh
  validation and locked test remain unevaluated and no global certificate is
  claimed.
- Made the buffered-contraction CVaR tail fraction explicit and froze the final
  one-bad-point calibration at 1%, after the 10% tail diluted the only remaining
  grid-63 violation across roughly 26 samples per batch.
- Recorded the max-loss calibration diagnosis and froze one final transform-only
  capacity adjustment from `rho=0.8` to `rho=0.9`; the gain trust region remains
  unchanged and the normalized Jacobian keeps a strict positive lower bound.
- Added an evaluation-only checkpoint path so fresh validation and locked test
  cannot silently take an optimization step after the final configuration is
  frozen.
- Froze and implemented the direct error-fiber multigrid route at `nu=0.005`:
  one physical four-mode gain and one low-mode conditional invertible transform
  are shared by grids 31, 63, and 127; the trainer differentiates the actual
  Allen--Cahn error dynamics, removes the structurally incompatible full-target
  defect from the objective, enforces invertibility by hard spectral projection,
  and keeps locked-test, gain-only, fixed-LMI, and four-sensor controls separate.
- Added an invertibility-preserving error-scale parameter and pre-registered a
  calibration-only capacity screen after the first full pilot saturated both
  the residual spectral budget and the gain trust region; formal validation is
  reserved for a fresh split after capacity selection.
- Corrected the missing `R = diag(-1,-1,1,1)` change of basis when moving the
  LMI gain and metric from NumPy `eigh` modes to the fixed physical sine basis;
  added hard regressions that the lifted gain and contraction rate reproduce
  the original LMI design before any network-capacity conclusion is allowed.
- Re-ran the corrected original-capacity pilot: one seed passed structure,
  direct-dynamics collocation, trajectory, and online gates on grids 31, 63,
  and 127. Cancelled the unnecessary capacity expansion and moved formal
  validation from the consumed calibration seed 1801 to fresh seed 1851.
- Recorded the first fresh-validation outcome: all three seeds failed only the
  grid-63 random-collocation worst margin while passing every trajectory,
  structure, and online gate. Reclassified seed 1851 as calibration and froze
  a gain-versus-transform capacity attribution screen before any new split.
- Completed the exact-commit RTX 2060 formal run and capacity attribution.
  Every seed passed structure, online, all trajectory, and grids 31/127
  collocation gates, but each missed one of 2048 fresh grid-63 low-mode
  collocations. None of the transform-only, gain-only, or joint capacity
  expansions met the frozen positive-buffer repair gate, so locked test stayed
  untouched and the route ended as a strict validation failure.
- Froze the next fresh-split joint experiment around the LMI transform:
  `T_phi(u,e)=T0[e+g_tilde(u,e)-g_tilde(u,0)]`, retaining the full nonlinear
  target and exactly four losses while enforcing global invertibility in
  `T0`-normalized coordinates.
- Froze a validation-only four-mode frequency audit for the three nonlinear-target
  checkpoints: decompose target defect and actual contraction power into `Pi_4`
  and `I-Pi_4` components, evaluate the projected-damping target counterfactually,
  and keep the locked test untouched.
- Implemented the checkpoint frequency-audit entry point with exact formal-metric
  replay, orthogonal defect/power additivity checks, per-case attribution,
  projected-target counterfactuals, and pre-registered pooled decision rules.
- Completed the exact-checkpoint RTX 2060 frequency audit. About 95% of learned
  target-defect loss and 92% of failed-sample negative contraction-power burden
  came from `Pi_4`; projected high-frequency damping improved counterfactual RMS
  by only about 2.1%, so the audit rejected it as the primary next repair.
- Implemented the frozen nonlinear-target joint trainer at `nu=0.005`, three
  sensors, and `n=31`: a spectrally projected conditional residual transform,
  a hard-trust-region constant gain, the exact nonlinear Allen--Cahn target,
  full-chain-rule dynamics, exactly four normalized losses, fresh split seeds,
  validation gates, locked test evaluation, and per-seed checkpoints.
- Reapplied the hard gain and transform projections after float64 validation
  conversion, preventing harmless float32-to-float64 singular-value drift from
  producing a false spectral-bound failure.
- Completed the corrected exact-commit RTX 2060 three-seed formal run. All
  structural invertibility checks and online-error gates passed, but all seeds
  failed the error-nonlinearity, worst-contraction, and target-defect progress
  gates; the locked test therefore remained unevaluated. Recorded the result and
  the baseline-preconditioned nonlinear-transform follow-up hypothesis.
- Drafted the next R5 joint-training plan at `nu=0.005`, three sensors, and
  `n=31`: replace the unjustified linear target with an analytically contractive
  nonlinear Allen--Cahn target, replace `P(u)e` with a spectrally bounded
  conditional invertible residual network, and retain exactly four normalized
  training objectives with validation-locked acceptance gates.
- Completed the exact-commit RTX 2060 `nu=0.005` joint multigrid run. Joint
  `B+T_phi` training reduced validation dynamics-defect RMS on grids 31, 63,
  and 127, but every grid failed the frozen worst-contraction, 25%-RMS, and
  online-no-regression gates; defect and worst margin also worsened with mesh
  refinement, so the mesh-robust classification is false.
- Froze a thesis-focused multigrid follow-up at `nu=0.005`: train only the native
  joint `B+T_phi` model independently on grids 31, 63, and 127, retain fixed LMI
  designs only as per-grid references, and compare validation dynamics and safety
  trends without test evaluation or early grid stopping.
- Added a dedicated `nu=0.005` multigrid runner that restricts every reused design,
  target, rollout, and audit helper to the single frozen viscosity; it trains only
  the native joint model on all three grids and emits per-grid gates plus an explicit
  mesh-trend classification.
- Froze the R5 three-sensor dynamics-joint experiment: six fixed/learned `B`/`T_phi`
  ablations, research-plan stable and continuous-defect losses, an ODE-inspired
  input-direction ablation, validation-only selection, and explicit gates for the
  incremental value of `T_phi` and joint training.
- Implemented the frozen six-row ablation, normalized defect-tail and direct-contraction
  audits, Jacobian/zero-fiber checks, validation-only model selection, locked test/noise
  evaluation, multi-grid expansion gates, checkpointing, and a dedicated command-line
  entry point for exact-commit RTX 2060 execution.
- Completed the exact-commit RTX 2060 coarse-grid run. Learned `T_phi` materially reduced
  held-out target-dynamics defect relative to gain-only training, but the selected joint
  model missed the strict worst-sample contraction, fixed-baseline RMS-improvement, and
  joint-synergy gates; test and finer grids therefore remained locked.
- 建立实验工程。
# Changelog

## 2026-08-24 — Pre-register low-mode adversarial repair

- Added a targeted repair plan using deterministic per-epoch resampling,
  constrained low-mode adversarial search on grid 63, and a positive contraction
  margin buffer.
- Added the checkpoint-initialized repair runner, compact-domain projection,
  buffered CVaR objective, and regression tests for the new training path.
- Recorded the failed first calibration without consuming the fresh validation
  split, then added fixed-pool replay, all-grid adversaries, teacher anchoring,
  and a transparent neighborhood replay of the consumed grid-63 bad point.
- After the second calibration repaired the original point but created other
  low-mode failures through excessive state conditioning, froze the inherited
  condition branch and added cosine learning-rate decay for the final
  pre-validation calibration.
- Added monotone checkpoint-capacity expansion so the common-LMI diagnosis can
  be followed by one controlled rho/gain-domain repair without changing the
  inherited checkpoint function.
- Corrected the min-max implementation to accumulate discovered adversarial
  constraints instead of replacing them at every refresh.
- Completed the exchange step by re-evaluating the full adversarial memory and
  training on its current worst active constraints instead of random history.
- Enabled the condition branch only after the frozen-condition active-set
  plateau demonstrated that a single near-state-independent error geometry was
  insufficient.
- Allowed repair checkpoints to seed a monotone follow-up capacity stage after
  the conditional model saturated its rho budget while leaving gain headroom.
- Added explicit mining of the consumed calibration split's current worst
  points for the final targeted fit; mined points are recorded as training-only
  and excluded from subsequent validation claims.
- Kept the PDE, sensors, grids, transform architecture, invertibility bound, and
  locked-test policy unchanged.

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
- Completed the CPU matrix/nonlinear gate and the frozen RTX 2060 joint-training screen. Five evenly
  distributed sensors passed all nine global semidiscrete certificate checks and nonlinear/noise
  validation. The original two sensors were linearly stabilizable with a general output injection, but
  every trained configuration failed both the positive-worst-contraction and nu=0.005 online-
  no-regression gates, so multi-grid expansion and test evaluation remained locked.
- Added a fixed-total-observation-length comparison for three and four sensors. The experiment freezes
  geometry selection on coarse-grid linear diagnostics, audits the three-sensor rank obstruction and
  transformed finite-trajectory contraction separately, selects the smallest qualifying four-sensor
  mass-adjoint gain, and unlocks test trajectories only after all validation gates pass.
- Completed the deterministic three/four-sensor study and an exact independent reproduction. Four
  interior sensors with total observation length 0.20 and mass-adjoint gain 0.50 passed all nine global
  semidiscrete margin checks. Three sensors cannot pass that global rank gate at nu=0.005, but a general
  modal injection passed every declared validation and locked-test transformed-contraction, online, and
  noise gate without GPU training.

- Added the R5 direct transformed-error contraction audit, checkpoint replay, contraction-aware joint
  loss, finite-sample worst-margin diagnostics, and pre-registered GPU training screen.
- Completed the RTX 2060 direct-contraction screen: the selected weight-10 model improved the worst
  validation rate from -0.4810 to -0.4043 while preserving invertibility and online error, but did not
  obtain a positive finite-sample validation margin.

## Unreleased

- Fast-forwarded the verified R5-G independent generalization and R5-H state-distribution OOD audits
  into the main direct-contraction experiment while preserving their original commits and result hashes.
- Reduced the active experiment surface to the direct-fiber training, adversarial repair, independent
  generalization, and state-OOD chain. Superseded route-specific designs, entry points, tests, and the
  early direct-contraction report were moved to a recoverable experiment-local trash directory; their
  independent archived experiment histories remain unchanged.

- Added the frozen R5-H evaluation-only state-distribution OOD matrix for the existing `nu=0.005`,
  three-sensor checkpoint. The audit separates the prior IID generalization conclusion from high-frequency
  truth/error, localized-pulse, multiple-interface, and large-initial-error robustness envelopes.
- Completed the RTX 2060 R5-H matrix. All sampled contraction margins remained positive and 14 of 15
  OOD cells passed both decision levels; the sole miss was the pre-registered relative-baseline terminal
  median gate for mode-9--12 initial error, while mode-13--16 error passed because of stronger diffusion.

- Implemented the pre-registered T0-preconditioned conditional invertible residual transform, fresh
  train/validation/test splits, normalized-coordinate nonlinearity gates, and a dedicated joint-training
  entry point while retaining the complete nonlinear target and the original four-term objective.
- Completed the three-seed RTX 2060 run: every seed passed contraction, target-defect, invertibility,
  and online gates, but all failed the frozen normalized nonlinear-in-error gate, so the locked test
  remained untouched.
- Added the R5 low-frequency certificate mode with grid-consistent physical sine projections,
  projected online correction, and separate high-frequency tail/coupling audits.
- Completed the R5 low-frequency/tail three-grid run and independent replay: the high-frequency
  tail remained small, while every low-frequency stability-margin gate failed on validation.
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
- Added state-dependent stable targets that retain either the Allen--Cahn nonlinear increment or its
  current-state Jacobian, with matched conservative decay, split-step target integration, and a
  pre-registered comparison against the existing fixed linear target.
- Completed the RTX 2060 state-dependent-target screen; both nonlinear targets slightly improved
  online reconstruction but increased held-out dynamics defect, so the pre-registered expansion gates
  failed and the route stopped before multi-grid training.
- Added the pre-registered R5 partitioned local-certificate definitions, strict worst-sample stability
  margin, physical overlap neighborhoods, and trajectory-transition audit.
- Added checkpoint replay tooling for per-region defect, contraction, invertibility, online-error,
  physical-overlap, and trajectory-switching audits.
- Completed the RTX 2060 audit on all 48 validation trajectories: no pre-registered region attained a
  positive strict local margin, and the declared horizon contained no phase-dominated samples.
