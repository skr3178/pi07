# Matched 200-update development pilot

Completed 2026-10-29. **The implementation trains, but this short run does not
show useful goal conditioning.** Decision: investigate with a longer matched
budget, then test live generated goals on development starts.

## Fixed setup

A and B start with exactly identical local SmolVLA weights and trainable modules
(99,880,992 parameters). Both receive the same 1,600 observations, action targets,
full task text, flow noise/time, 200 optimizer updates and batch size 8. The VLM
is frozen and verified unchanged; the action expert/projections and state
projection train. AdamW uses 1e-4 after 20 warmup updates, betas 0.9/0.95,
gradient clipping 10 and BF16 autocast.

A always masks both goal slots. B receives real future goals on **370/1,600 =
23.125%** of observations, within the declared three-sigma range around 25%.
Both use the corrected padding loss and fixed four-slot attention layout.
The 450/50 episode split and all schedules were saved before training.

Held-out diagnostics use 100 fixed observations, 10 per task. This is an
adaptation holdout, not proof of data unseen during pretrained-model training.

## Results

Lower is better. Flow MSE uses fixed noise/time. Action MSE samples a full
50-step action chunk with ten denoising steps and scores the first 8 valid
steps in the checkpoint's normalized 7-D action space. Neither metric is
simulator success.

| Condition after 200 updates | Validation flow MSE | Sampled action MSE, first 8 |
| --- | ---: | ---: |
| A, goals dropped | 0.3890 | 0.4038 |
| B, real future goals | 0.4585 | 0.5176 |
| B, goals dropped | 0.3902 | 0.3816 |
| B, copies of current images | 0.4400 | 0.4689 |
| B, goals from a different task | Not measured | 0.5661 |

B's real future goals perform worse than dropping goals or copying current
images in this small diagnostic. Mismatched-task goals are worse again.
This does not identify a single cause: the model has seen only 370 examples
with goals, and changed token positions/attention and the goal representation
may also matter. It does not test ImageWAM, whose generated images are absent
from this pilot.

Both checkpoints saved and reloaded exactly. A/B initial state hashes, flow
noise/time hashes, example counts, update counts and trainable parameter counts
match. All optimization steps were finite and the frozen backbone stayed
unchanged. Peak allocated GPU memory was **4.709 GB for A / 4.714 GB for B**;
reserved memory was 4.853 GB. Update median excluding decode was
0.254 / 0.261 seconds. Training plus periodic validation took 66.8 / 89.0
seconds, excluding initial load, initial validation and checkpoint serialization.

## Files and next diagnostic

- [Manifest](manifest.json), [training report](report.json), [offline action report](offline_action_diagnostics/report.json).
- [A checkpoint](A/), [B checkpoint](B/), including original saved normalization
  processors and optimizer state. Load with `goal_policy.GoalSmolVLAPolicy`,
  not the upstream plain SmolVLA class. `goal_interface.json` describes the masks.
- [Source snapshot](source/); script hashes are in the manifests.
- [1,000-update extension](../goal_training_1000/): fresh matched A/B training
  from the same base, with a new predeclared shared schedule and the same
  validation observations. Chosen after this development result; no final
  benchmark starts have been used.

From the project root, a fresh pilot can be run with:

```bash
/media/skr/storage/envs/pi07-lerobot/bin/python -u pi07_code/train_goal_pilot.py \
  --output pi07_code/artifacts/NEW_RUN --steps 200
/media/skr/storage/envs/pi07-lerobot/bin/python -u pi07_code/evaluate_goal_pilot.py \
  --run pi07_code/artifacts/NEW_RUN
```

Both scripts refuse to overwrite an existing run/diagnostic directory.
