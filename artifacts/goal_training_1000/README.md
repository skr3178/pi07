# Matched 1,000-update development run

Training completed on 2026-10-29. This extends the [200-update pilot](../goal_training_pilot/README.md), where
real future goals did not improve the policy's offline action error. It is a
development experiment, not the final benchmark.

## Predeclared settings

- Both arms restart from the same base SmolVLA checkpoint. Each receives 1,000
  updates, batch size 8, and 8,000 examples from the same saved schedule.
- B gets goals on 1,994/8,000 examples (**24.925%**); A always masks them.
- The trainable modules, optimizer, 20-update warmup, flow noise/time seeds,
  corrected padding loss and fixed attention layout match the 200-update pilot.
- The validation schedule is verified array-for-array identical to the short
  pilot: 100 observations from the saved 50-episode holdout, 10 per task.
- The new shared training schedule is fixed before this run. A's initial
  observation prefix matches the short run; B's goal-presence/future schedule
  differs because random draws follow the larger observation sample.
- Validation is recorded every 50 updates. The final 1,000-update checkpoints
  are the planned checkpoints for the development rollout; selection is not
  based on the lowest recorded validation value.

[manifest.json](manifest.json) pins choices and hashes. [report.json](report.json)
contains the live status, all updates, validation curves, memory, frozen-weight
checks and exact save/reload checks. [source/](source/) preserves the training
and offline evaluation code used for this run.

## Completed training measurements

| Final checkpoint condition | Held-out flow MSE |
| --- | ---: |
| A, goals dropped | 0.3504 |
| B, real future goals | 0.3760 |
| B, goals dropped | 0.3589 |
| B, copies of current images | 0.3729 |

Both arms completed all 1,000 finite updates. Initial parameter hashes,
trainable counts, observation/target schedule, flow noise/time hashes and update
budgets match. Frozen backbone hashes are unchanged and both saved checkpoints
reload exactly. Peak allocated memory: **4.709 GB A / 4.714 GB B**. Training plus
periodic validation: **342.9 seconds A / 444.5 seconds B**, excluding initial
load/validation and checkpoint serialization.

The goal-conditioned gap is smaller than in the short pilot, but these flow
losses still do not demonstrate a benefit from real future images. The
[validation curve](validation_curve.png) shows every measured checkpoint.
The planned final checkpoint is used even though step 950 has a lower loss;
there is no post-hoc best-checkpoint selection.

## Offline sampled actions

The [offline action report](offline_action_diagnostics/report.json) is complete.
It samples ten-denoising-step action chunks on the same 100 observations and
noise draws, then excludes episode padding and scores the real seven action
dimensions in normalized units.

| Condition | MSE, first 8 actions | MSE, full valid 50-step chunk |
| --- | ---: | ---: |
| A, goals dropped | 0.2892 | 0.3313 |
| B, real future goals | 0.3403 | 0.3575 |
| B, goals dropped | 0.2882 | 0.3411 |
| B, copies of current images | 0.3239 | 0.3563 |
| B, goals from a different task | 0.3546 | 0.3722 |

All five conditions produced finite actions. More training improved the action
diagnostic, but real future goals still do not beat dropped or copied-current
goals. This result predates using ImageWAM in the policy loop, so a generated
image mismatch cannot explain this particular gap. Action error against one
demonstration is not definitive evidence of closed-loop success or failure.

The [live development comparison](../goal_closed_loop_dev/README.md) is complete:
A, B dropped and B generated each scored 0/6 on the same development starts,
with no runtime errors. The final 100-start benchmark remains reserved. The
next diagnostic addresses baseline readiness and goal conditioning; all jobs
have finished.

## Commands

```bash
/media/skr/storage/envs/pi07-lerobot/bin/python -u pi07_code/train_goal_pilot.py \
  --steps 1000 --output pi07_code/artifacts/goal_training_1000
/media/skr/storage/envs/pi07-lerobot/bin/python -u pi07_code/evaluate_goal_pilot.py \
  --run pi07_code/artifacts/goal_training_1000
/media/skr/storage/envs/pi07-lerobot/bin/python -u pi07_code/run_goal_development.py \
  --run pi07_code/artifacts/goal_training_1000
```

Existing manifests/results are protected against overwrite. Choose a fresh
output directory for a repeat. Do not rerun the training command over this run.

The simulator test uses only development starts 0/1 on simulator tasks 0/4/8.
The [reserved final 100 starts](../libero10_evaluation_design/README.md) remain
separate. Neither adaptation validation nor those reserved starts establish
data unseen by the released pretrained models.
