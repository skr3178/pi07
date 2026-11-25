# π0.7 on LIBERO: implementation and evidence report

Status: **in progress**, 2026-11-04. The active [plan](IMPLEMENTATION_PLAN.md) lists live jobs and pending work. A negative result is retained as evidence; pending experiments are not treated as completed.

## What is implemented

The local stack uses the downloaded LIBERO-90 SmolVLA checkpoint, two current camera views, eight-dimensional state and seven-dimensional end-effector actions. The action expert/state projection are trained; the visual-language backbone stays frozen. ImageWAM supplies two generated goal views from the actual current cameras/state and task or oracle-subtask text.

| Mechanism | Implementation | Evidence and limits |
| --- | --- | --- |
| Subtask text | Dataset phase labels and runtime predicate oracle | Ten LIBERO-10 plans checked; three uncertain demo episodes omit phase labels. Privileged-state oracle, not autonomous language planning. |
| Visual goals | Two masked goal camera slots; real future frames offline, ImageWAM online | Real/copy/wrong/off controls saved. Runtime never receives demonstration future frames. Useful future content is not yet established. |
| Metadata | Duration, quality 1–5, sparse mistake flags with provenance | Curated demo/rollout mixture and two-update integration verified; full-budget pair queued. Quality/mistake judgments are provisional assistant reviews. |
| Control mode | Fixed `ee` prompt and end-effector interface | One controller/robot; no switching or transfer claim. |
| Dropout | 25% goal presence; 30% conditional subtask, 15% all-metadata, 5% field dropout | Sample masks/gradients checked; profiles trained. Controlled text/image factorial uses zero conditional text dropout. |
| Generated training images | Verified cache routing and real/generated replacement | Two actual ImageWAM goals used in smoke training. Larger 500-observation cache and full-budget comparison remain pending. |
| Metadata CFG | Same-noise positive/negative flow fields; metadata removed only in negative branch | Beta=0 and identical-branch numerical checks exact; live smoke passes. Behavior sweeps pending. |

## Matched training results

Normalized first-eight action MSE on saved held-out demo observations; lower is better. These are not simulator success rates. All runs below completed 1,000 updates with exact checkpoint reload and unchanged frozen backbone.

| Dataset | Profile | Initial MSE | Final MSE | Same model, goals off |
| --- | --- | --- | --- | --- |
| LIBERO-10 | task | 0.6197 | 0.5151 | — |
| LIBERO-10 | subtask | 0.6168 | 0.5139 | — |
| LIBERO-10 | goal | 0.6800 | 0.5661 | 0.4934 |
| LIBERO-10 | subtask_goal | 0.7478 | 0.5864 | 0.5103 |
| LIBERO-10 | full | 0.7752 | 0.5868 | 0.5171 |
| LIBERO-Goal | task | 1.1638 | 0.6230 | — |
| LIBERO-Goal | visual | 1.1755 | 0.6300 | 0.6260 |

LIBERO-10 combined: real goals **0.5864**, copied-current **0.5848**, wrong-task **0.5872**, goals off **0.5103**. LIBERO-Goal visual: real **0.6300**, copied-current **0.6333**, wrong-task **0.6329**, goals off **0.6260**. These controls do not establish a useful future-image effect.

[LIBERO-10 matched report](artifacts/section_v_matched_summary_20261104/README.md) · [LIBERO-Goal matched report](artifacts/section_v_goal_matched_summary_20261104/README.md).

## Completed historical live comparison

The older 2,000-update LIBERO-90 models were evaluated on 20 paired development starts each: reference **0/20**, task-only A **2/20**, B goals off **0/20**, B generated **4/20**, B copied-current **4/20**. Generated and copied-current succeeded on exactly the same four starts. This does not attribute the apparent improvement over A/off to future-image information. That older B was trained with goals always present, so its goals-off condition is distribution-shifted.

[Audited paired results](artifacts/libero90_paired_controls_20261104/README.md): 80 cross-condition full-reset comparisons, 599 recorded generation inputs, and 593 copied-current pairs verified.

The fresh Section V live comparison, familiar-instruction Goal live comparison and metadata behavior sweep are still in progress or queued. Their final reports must be inspected before any conclusion.

## Data and semantic checks

The LIBERO-Goal FastWAM data uses `0 = close, 1 = open`; this was converted to simulator `+1 = close, −1 = open` before any Goal policy training. Thirty decoded observations, their 50-action chunks and 520 padded targets were checked, with finger-motion sign evidence. Arm actions/state are preserved. [Correction audit](artifacts/libero_goal_data_check_v2_20261104/README.md).

The curated metadata study combines six reviewed demos with twelve collected episodes (5,698 rollout transitions, two successes). All 18 episodes have provisional assistant quality ratings. Only 64 individual frames have mistake/no-mistake review; unknown frames are omitted. The fixed 8,000-sample training schedule has 45 visible mistake=true and 13 mistake=false prompts, with quality values 1/2/4/5; no runtime quality-3 request is supported. Physical rollout starts are grouped across perturbed/clean variants. Demo physical-start overlap remains unverified. [Mixed-data audit](artifacts/metadata_mixture_check_20261104.json).

The initial generated cache has 20 predictions: 14 judged plausible near continuations, five uncertain, one rejected. These are visual judgments, not proof of semantic completion or physical achievability. A separate [live four-image review](artifacts/section_v_l10_development_20261104/review/README.md) finds plausible arm motion but no completed basket placement in those inspected goals.

## Remaining work before completion

1. Finish and audit the running/queued simulator and metadata experiments.
2. Generate the 500-observation training cache, review the predeclared 50-observation stratified sample, and run a full-budget generated-augmentation comparison with its actual mixture fraction reported.
3. Freeze the declared checkpoint/runtime choices and run the reserved final LIBERO-10 cohort: 100 starts per condition. Final guard checks pass; no final episodes have run.
4. Consolidate success, progress, paired uncertainty, resource cost and unsupported claims; perform the requirement-by-requirement completion audit.

## Interpretation limits

ImageWAM predicts 16 training transitions, whereas the paper trains a world model for segment endpoints. LIBERO-10 instruction steering is unproven; scene-order bias is strong. The policy is much smaller and mostly frozen, training uses one model seed and a limited update budget, filtered dataset frames do not certify physical elapsed time, and pretrained-data overlap is not excluded. Metadata labels lack independent human agreement. Synchronous generation pauses simulation. No claim is made for real-time robotics, multi-robot transfer, novel composition or reproduction of π0.7’s reported success rates.

[Run guide](REPRODUCTION.md) · [Environment/source records](artifacts/reproduction_environment_20261104/).
