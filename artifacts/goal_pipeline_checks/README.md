# Goal-conditioning implementation checks

Completed 2026-10-29. Decision: **continue to matched development training**.
These checks establish that the implementation works; they do not establish a
policy benefit from generated images.

## Measurements

| Check | Result |
| --- | --- |
| Future pairs audited | 10,000; 0 cross-episode pairs |
| Goal-present frequency | 2,509/10,000 = 25.09% |
| Segment-end fraction among present goals | 24.07% |
| Camera identity checked against decoded source | 10/10 pairs, one per task, exactly equal |
| Split | 450 training / 50 validation episodes, 5 validation episodes per task |
| Strict pretrained checkpoint load | Passed; 450,046,176 parameters |
| Trainable parameters | 99,880,992; VLM/vision backbone frozen |
| Changing masked goal pixels: maximum action change | 0.0 |
| Changing another batch member's goal mask: maximum action change | 0.0 |
| Current observation attention into goals or text | 0 allowed edges |
| Adding 1,000 to padded action targets: loss change | 0.0 |
| Finite gradient tensors on first backward | 155/155 |
| Fixed-batch learning, 30 updates | Loss 1.84547 → 0.00743146, a 99.60% reduction |
| Update latency, median / p95 | 0.0936 / 0.1007 seconds, batch size 2 |
| Peak allocated / reserved GPU memory | 2.754 / 2.802 GB |

The valid-goal condition changed mean normalized actions by 0.2633 before
training. This shows that the input can affect the policy, not that the effect
is useful or semantic. The fixed-batch check uses fixed noise/time and its
weights were discarded; its loss reduction is not validation performance.

## Implementation choices

- [goal_policy.py](../../goal_policy.py) adds main/wrist goal fields and explicit
  validity masks using the existing shared image encoder. Both A and B use the
  same four image slots and attention layout. Current observations attend only
  to observations; goals and text also see observations; state and actions see
  the available prefix. This changes the original checkpoint's attention
  convention in **both** arms, so the previous unmodified reference pilot is
  not training baseline A.
- An initial optimization removed goal slots when every sample dropped goals.
  The deterministic action comparison changed by 0.0156884. The exact numerical
  cause was not isolated. That optimization was removed; the fixed-layout
  implementation passes both invariance checks exactly. The initial failure is
  retained in [failed_variable_slots/](failed_variable_slots/).
- The training wrapper consumes the dataset's `action_is_pad`, canonicalizes
  padded targets before flow noise, and averages loss only over valid timesteps
  and the real 7 action dimensions. This corrects the pinned upstream padding
  key mismatch and excludes the unused dimensions of its 32-D internal action.
- [goal_data.py](../../goal_data.py) samples frames within episodes, with 25%
  goal presence. Present goals use a 25% segment-end / 75% bounded-future mixture.
  The bounded future is uniform over 0–40 remaining dataset frames. Task 3's
  first one-frame label artifact is merged into its next label in memory in
  50 episodes; source trajectories remain intact.

The split is verified only by episode. The dataset does not expose simulator
initial-state IDs, so physical-start grouping is **unverified**. Both released
models have prior LIBERO training. Neither this split nor this diagnostic proves
generalization to starts unseen by pretraining. The 10-fps dataset labels are
not a verified mapping from 40 frames to four physical seconds.

The [offset audit](offset_audit.json) separately records bounded-future and
segment-end offsets. Segment ends can be more than 40 frames away; the 40-frame
limit applies only to the ordinary-future branch. Zero-offset goals are allowed
by the declared inclusive sampling interval.

## Reproduce and resume

[report.json](report.json), [episode_split.json](episode_split.json) and
[audit_schedule.npz](audit_schedule.npz) hold the measurements and exact choices.
The next stage is [the matched training pilot](../goal_training_pilot/).

```bash
/media/skr/storage/envs/pi07-lerobot/bin/python -u pi07_code/check_goal_pipeline.py
```

This check overwrites its current report; archive it before a comparative rerun.
