# π0.7 on LIBERO: active implementation plan

Created 2026-11-04. This is the current resume point for the user-requested
implementation goal. Historical plans and results remain available in
[reconstruction.md](reconstruction.md), [todo.md](todo.md), and
[run_scorecard.md](run_scorecard.md). Earlier next-step instructions in those
documents are historical when they disagree with this plan.

## Scope and completion criteria

Recreate the mechanisms in Section V of the local [paper](pi_07.pdf) using
SmolVLA, LIBERO and ImageWAM on the available 24 GB GPU. The deliverable is
reproducible data preparation, training, simulator evaluation, artifacts and
an evidence-based report. A negative result is a valid result; a smoke test
or an untested code path is not a completed reproduction.

| Paper mechanism | Local implementation | Remaining work / declared approximation |
| --- | --- | --- |
| V-A subtask text | Audited subtask IDs, text adapter and simulator-predicate oracle | All ten plans checked; matched 1,000-update text training complete, live comparison running. This is privileged-state assistance, not autonomous language planning. |
| V-B multiview goals | Two goal camera slots, real future images; ImageWAM worker | Validate meaningful outcomes and run live generated-goal comparisons. ImageWAM predicts 16 transitions ahead; it is not the paper's segment-end BAGEL model. |
| V-C metadata | Length, quality and mistake schema with provenance | Twelve rollouts collected (5,698 transitions); curated six-demo mixture has provisional assistant quality labels and 64 reviewed mistake/no-mistake frames. Paired metadata training remains pending. These are not independent human labels. |
| V-D control mode | Fixed end-effector tag | One robot/controller only; no joint-mode switching claim. |
| V-E dropout | 25% goals, 30% conditional subtask dropout, 15% whole-metadata and 5% per-field dropout | Train and compare profiles, not merely validate masks. A controlled text × image experiment must hold text availability fixed. |
| VI-C generated training images | Cache builder, provenance checks and real/generated mixture implemented | 20 real predictions reviewed: 14 plausible near continuations, five unresolved, one rejected. Two-update actual-generated-image training passes. A 500-observation training cache is queued for a full-budget augmentation comparison; review and efficacy training remain pending. The paper does not specify a 50% generated/real ratio. |
| VII runtime | Synchronous generation, refresh timer, subtask-triggered refresh and metadata-only CFG | Zero-guidance and identical-branch numerical identities pass; two live eight-step metadata checks pass. Goal generation with the live oracle still needs evaluation. No asynchronous real-time, attention-tree optimization or RTC claim. |

No reproduction of the paper's model scale, diverse robot fleet, history
encoder, internet-scale training, novel composition or cross-robot transfer is
claimed. LIBERO-Goal familiar instruction control is a separate experiment
from LIBERO-10 usual-continuation guidance.

## Local assets reviewed

| Asset | Location | Decision |
| --- | --- | --- |
| LIBERO-10, 500 demonstrations / 138,090 frames | `data/libero_10_subtask` | Main data. Existing 450/50 episode split; physical-start and pretraining independence are not certified. |
| LIBERO-Goal image data | `/media/skr/storage/datasets/LIBERO-fastwam/lerobot_v30/libero_goal_no_noops_lerobot` | 433 episodes, 52,895 frames, ten tasks; frozen 383/50 episode split. Corrected gripper targets with `1 - 2 * raw` before any Goal training; 30 decoded examples and all chunk targets/padding verified. See the v2 action-convention audit. |
| LeRobot / LIBERO | `vendor/lerobot`, `vendor/lerobot-libero`; `/media/skr/storage/envs/pi07-lerobot` | Working environment, exact reset-state checks previously completed. |
| Original LIBERO SmolVLA | `/media/skr/storage/models/smolvla_libero` | Historical reference; training overlap limits generalization claims. |
| LIBERO-90 SmolVLA | `/media/skr/storage/models/smolvla_libero_90_pi07` | Current reference and starting checkpoint; local adapter retains source weights/normalizers. Training-subset provenance is weaker than a certified unseen-data claim. |
| Vanilla SmolVLA | `/media/skr/storage/models/smolvla_base` | Also downloaded. Its robot interface needs adaptation; not substituted into the current comparison. |
| Saved 2,000-update A/B-near/B-end | `/media/skr/storage/checkpoints/pi07_goal_ab/l90v2_{A,Bnear,Bend}` | Reuse for live tests; load **AltGoalPolicy**, matching the actual training layout. |
| ImageWAM LIBERO / FLUX.2 / Qwen3 | `/media/skr/storage/models/imagewam`; `ImageWAM`; `/media/skr/storage/envs/imagewam-venv` | Ready. Qwen prompts are encoded once then the encoder is unloaded. |
| SuSIE | `../susie` | Useful reference implementation; released example targets BridgeData, not a verified local LIBERO generator. No replacement needed yet. |
| Interactive World Sim / Instant Policy / Learning Thousand Tasks | Sibling repositories | Different interfaces/domains; do not substitute without a separate compatibility experiment. |
| BAGEL / Cosmos alternatives | Earlier reports | Prior BAGEL probe failed to give useful robot goals. Cosmos integration is not a validated local replacement. |

GPU checked 2026-11-04: RTX PRO 4000 Blackwell, 24,467 MiB; approximately
414 MiB used before this work. Free disk approximately 105 GB workspace,
82 GB external storage. GPU access needs execution outside the filesystem
sandbox; the read-only check succeeded. No new model downloads are planned.

## Evidence carried forward

- Original six-start development comparison: A, B dropped and B generated
  each 0/6. Functional integration only; not a suite-wide null result.
- Latest LIBERO-90 offline evaluation: A 0.4205; B-near real 0.4103,
  copied-current 0.4041, wrong-task 0.4227; B-end real endpoint 0.4302.
  These are normalized first-eight action MSE on 100 observations / 43
  episodes, one training seed. They do not demonstrate a useful goal effect.
- ImageWAM can distinguish familiar LIBERO-Goal instructions, but our
  LIBERO-10 boundary test did not establish reliable redirection. Generated
  pictures are not automatically semantic endpoints or physically achievable.
- All five Section V profiles completed 1,000 updates. Held-out first-eight
  normalized MSE: task **0.5151**, subtask **0.5139**, goal **0.5661**,
  subtask+goal **0.5864**, full **0.5868**. Exact reload and unchanged frozen
  backbone verified. These are one-seed development results.
- For the combined checkpoint, real goals **0.5864**, copied current **0.5848**,
  wrong-task goals **0.5872**, goals off **0.5103**. Useful goal content is not
  established. [Matched report](artifacts/section_v_matched_summary_20261104/README.md).
- Provisional quality/mistake labels now exist for a small curated mixture;
  mixed-quality training remains pending. Speed labels are retained episode
  lengths, not certified wall-clock durations.

## Execution order and quantitative gates

1. **Trustworthy live baseline and generated-goal evaluation (completed development round).**
   Use the saved reference/A/B-near policies on identical task names, full
   simulator states and random seeds; all ten tasks, development starts 0/1.
   Compare reference, A, B dropped, B generated and B copied-current. Record success, steps,
   BDDL predicate progress, raw actions, clipping, generated inputs/images,
   latency and memory. Gate: zero runtime errors, finite 7-D actions, full
   initial-state maximum difference ≤1e-8. Report per-task outcomes and paired
   changes, not just aggregate success. Final starts 2–11 remain untouched.

2. **Policy and subgoal diagnosis.** Verify LIBERO-90 wrapper predictions at
   update zero; fit 32 fixed training observations while monitoring both
   flow loss and sampled first-eight/full-chunk error. Record learning-rate,
   warmup and gradient clipping. Separately inspect current/near/endpoint/
   generated camera pairs with explicit semantic-completion ratings and
   annotation uncertainty. Existing three ambiguous two-pot episodes need
   explicit omission/unknown handling. A null goal effect is a diagnostic,
   not proof that image conditioning is impossible.

3. **Finish the reusable Section V training interface.** Select base/layout
   explicitly, retain identical schedules across profiles, save update-zero
   evaluations and restartable checkpoints, and support generated-image
   augmentation. Record source hashes, input provenance, label coverage,
   prompt/dropout distributions, real/generated ratio and physical-time
   assumptions. Cache entries must not cross episode splits or silently use
   different prompts/observations.

4. **Run declared prompt ablations.** Task, subtask, goal, combined; same
   checkpoint/data/actions/optimization budget. Separate a controlled
   factorial from the paper's conditional dropout recipe. Subtask inference
   uses a declared human/oracle/controller source, never demo frame timings.
   Evaluate real goals offline; generated/copy/off goals online. Establish
   learning and runtime behavior before scaling steps or seeds.

5. **Mixed-quality metadata and runtime requests.** Collect successful,
   failed, slow and recovered simulator behavior on matched tasks. Preserve
   raw observations/actions/events. Review quality and mistake annotations
   with provenance; report coverage and agreement. Implement metadata-only
   CFG and test its zero-guidance identity before behavior sweeps.

6. **Freeze and report.** Save configs/checkpoint hashes and evaluate the
   reserved final starts only after development choices are frozen. Report
   success, progress, uncertainty and cost; distinguish model-seed variation
   from evaluation-start variation. Record all unsupported paper claims.

## Current entry points

- `evaluate_libero.py`: checkpoint/layout-aware, matched simulator evaluation.
- `experiments/libero90_runtime_smoke.json`: eight-step, four-condition integration check.
- `experiments/libero90_development.json`: all-ten-task, two-start development evaluation.
- `train_section_v.py`, `section_v_data.py`, `section_v_prompt.py`: prompt training foundation.
- `imagewam_goal_worker.py`: actual-current-observation → multiview generated goal.
- `prepare_generated_goals.py`, `generated_goal_data.py`: cache generation, review sheets and verified training mixtures.
- `subtask_controller.py`: privileged-state oracle; insertion before closure, recovery when predicates become false again.
- `check_policy_training.py`: update-zero layout comparison and 32-observation fitting diagnostic.
- `analyze_libero_evaluation.py`: trajectory audit, task-level paired uncertainty and report generation.

## Measured implementation gates, 2026-11-04

- Four-condition runtime smoke complete: reference, A, B-off and B-generated;
  eight actions each, zero errors, identical full simulator resets. One live
  generated pair matched the trajectory state exactly. ImageWAM startup peak
  17.60 GB allocated; the text encoder was unloaded. This is not an efficacy test.
- Section V checks pass on 10,000 sampled pairs and ten decoded task examples.
  Episodes **177, 185, 189** now omit uncertain phase text; 89/10,000 sampled
  contexts have unknown subtask. Quality/mistake coverage is zero in this
  original demo-only check; the later curated metadata study is separate.
- New LIBERO-90/alternative-layout full-prompt one-update CPU smoke complete:
  finite forward/backward, unchanged frozen backbone, exact checkpoint reload.
  Added sampled-action evaluation at step zero; one example is not a baseline.
- Cache routing checks pass: current images and actions unchanged; camera
  routing correct; hidden goals not replaced; zero mixture selects none; five
  malformed provenance/input cases rejected. One real decoded observation plus
  synthetic colored images; fixtures are not training annotations or goals.
- Oracle controller maps all ten BDDL tasks and passes synthetic transition,
  regression and insertion-before-close checks. Three live 16-step checks pass;
  these establish interfaces, not task performance.
- LIBERO-90 plain/alternative layout at update zero (CPU, eight training
  observations): maximum action difference **0.01111**, RMS **0.001574**.
  Small but nonzero; do not claim bitwise equivalence or certify all contexts.

### Completed live result

The original 80 episodes plus 20 copied-current controls are audited together in
[libero90_paired_controls_20261104](artifacts/libero90_paired_controls_20261104/README.md).

| Condition | Successes | Mean maximum predicate fraction |
| --- | --- | --- |
| LIBERO-90 reference | 0/20 | 0.133 |
| Adapted A | 2/20 | 0.283 |
| B-near, goals off | 0/20 | 0.133 |
| B-near, generated goals | 4/20 | 0.308 |
| B-near, copied current images | 4/20 | 0.333 |

Generated and copied goals succeed on **exactly the same four starts**. This
does not establish a benefit from predicted future content. B-near was trained
with goals always present, so goal-off is also a distribution shift. The new
Section V profiles use 25% goal presence. The audit checks 80 cross-condition
full simulator resets, 599 generated input/image pairs and 593 copied pairs.
One model training seed and 20 development starts per condition are not a final
benchmark. Latency overlap is recorded; it is not an isolated speed comparison.
Original/resumed source snapshots and manifests are preserved.

### Training and interface diagnosis

- Both models fit 32 fixed training observations in 200 updates: A first-eight
  MSE **0.5987 → 0.0265**, B **0.8222 → 0.0255**. B copied-current is **0.0270**,
  wrong-task **0.0380**, dropped **0.0955** on this same training subset. Fitting
  works; this does not establish generalization or useful future-image content.
- GPU plain/alternative-layout first-eight difference at update zero: maximum
  **0.00825**, RMS **0.00199**. Small but nonzero; full-chunk maximum is larger.
- Immutable intermediate checkpoints and optimizer/RNG recovery implemented.
  Two-update CPU check: resumed and uninterrupted model bytes, final update and
  validation are identical. Future long jobs use the persistent launcher;
  launch it outside the sandbox, whose detached children do not persist.
- [Generated-image pilot](artifacts/generated_subtask_cache_20261104/README.md)
  retains raw and reviewed manifests, both cameras and per-image rationales.
  Ratings are assistant visual inspection, not independent human annotations.
- Metadata CFG follows the paper's **v_cond + beta × (v_cond − v_no_metadata)**
  convention: beta=0 means ordinary conditioned inference. Both numerical
  identities have maximum error zero on a real observation. The two-update
  checkpoint used for this check is not a behavior-trained metadata policy.

Dataset task IDs and simulator suite IDs have different orders. Match them by
full task text; all online logs retain the actual simulator instruction.

### Active work and next commands

- `artifacts/section_v_l10_{task,subtask,goal,subtask_goal,full}_1000_20261104`:
  all complete. The controlled factorial uses conditional text dropout zero;
  full uses the paper's 30% setting, so full vs combined alone does not isolate metadata.
- `artifacts/section_v_l10_development_20261104`: 100 paired live episodes
  running (A, subtask, combined off/copied/generated). Starts 0/1 only.
- `artifacts/section_v_goal_{task,visual}_1000_20261104`: matched LIBERO-Goal
  training complete with corrected gripper convention, unchanged frozen
  backbone and exact reload. Task MSE **1.1638 → 0.6230**; visual model real
  **0.6300**, copied-current **0.6333**, wrong-task **0.6329**, goals off
  **0.6260**. [Summary](artifacts/section_v_goal_matched_summary_20261104/README.md),
  [action audit](artifacts/libero_goal_data_check_v2_20261104/README.md).
- `data/libero10_metadata_mixture_20261104`: six reviewed demos plus twelve
  collected rollouts; use **annotations_v2.jsonl**. Both mistake values occur
  in the training split. `artifacts/metadata_mixture_schedule_20261104` fixes
  8,000 train samples and 30 validation observations. Compare combined vs full
  with the same 30% text dropout to isolate metadata. Provisional assistant
  labels and sparse mistakes limit interpretation.
- `artifacts/metadata_mixture_check_20261104.json`: 54 observations and 882
  padding positions checked; paired profiles have identical goals/subtask
  dropout. Training has 45 visible mistake=true and 13 mistake=false prompts,
  and quality values 1/2/4/5. Quality 3 is absent from training and must not be
  requested in runtime sweeps. `metadata_training_smoke_20261104` passes two
  actual mixed-source updates and exact checkpoint reload.
- `artifacts/generated_training_smoke_20261104`: two actual ImageWAM goals
  routed through two CPU updates; finite training and exact reload. This is an
  integration check, not a generated-augmentation efficacy result.
- Job records and logs: `artifacts/jobs/`. Check `background_job.py status`
  outside the sandbox before restarting any job; saved `running` alone does
  not prove a process is alive.
- Jobs `section_v_metadata_{subtask_goal,full}_1000_20261104` are **queued**
  behind the current L10 live run and both Goal training jobs. They use the
  same 30% conditional text dropout and v2 annotations.
- `section_v_goal_offline_20261104`: complete, six image-content controls.
- `section_v_goal_development_20261104` waits for metadata training and Goal
  offline controls, then runs 100 episodes: original pretrained reference,
  matched task control, and visual policy off/copied/generated (20 each).
- `section_v_metadata_development_20261104` waits for that Goal live run,
  then runs nine conditions on three tasks × two starts: no-metadata control,
  metadata-dropped, quality 1/5, quality 5 with CFG beta=1.3, speed 200/500,
  mistake true/false. Requests are exploratory, especially sparse mistakes.
- `run_after_jobs.py` stops the chain if a dependency fails; a waiting job
  appears as running in its launcher, with dependency states in its log.
- `generated_training_cache_500_20261104` is queued behind metadata live
  evaluation. It will sample 500 training-only observations from the combined
  profile's visible-goal schedule, balanced across tasks. After generation,
  review a predeclared stratified sample and exclude known rejected entries;
  train a full 1,000-update augmentation arm with the same combined-profile
  settings. Report actual generated fractions and unreviewed coverage rather
  than equating the requested cache mixing probability with the overall ratio.
- `freeze_protocol.py` now gates final evaluation on completed, audited
  development conditions and immutable settings/checkpoint/runtime/simulator
  asset hashes. Ten guard checks pass. `section_v_l10_final_candidate.json`
  is **unfrozen**: A, combined-off, combined-copy, combined-generated, each
  on all ten tasks × starts 2–11 (100 per condition). No final episodes have
  run. Complete development and augmentation evidence before freezing.
- Next: inspect those jobs; analyze each completed live run with
  `analyze_libero_evaluation.py`; summarize each matched training pair with
  `summarize_section_v_runs.py`. Then freeze choices and evaluate reserved
  starts 2–11. ImageWAM startup needs about 17.6 GB, so do not overlap it with
  two policy trainings. [Run guide](REPRODUCTION.md).

Update this file with measured outcomes and the exact next command as work
continues. The active goal remains open until required work is completed or
the user changes its scope.
