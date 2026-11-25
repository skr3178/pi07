# LIBERO-90 live development evaluation

Completed 2026-11-04.

Same task/start pairs, seeds, 50-action predictions with eight executed, 8-step limits.
Generated goals use one ImageWAM call from current main/wrist images and state every 16 simulator actions.
Simulation pauses during inference. No demonstration future is supplied online.

| Condition | Successes | Mean maximum predicate fraction | Mean gain over initial state |
| --- | --- | --- | --- |
| reference | 0/1 | 0.000 | 0.000 |
| A | 0/1 | 0.000 | 0.000 |
| Bnear_off | 0/1 | 0.000 | 0.000 |
| Bnear_generated | 0/1 | 0.000 | 0.000 |

Verified 4 episodes, finite actions and paired full simulator resets. Audited 1 generated pairs; maximum generator/trajectory state difference 0.0.

See [analysis.json](analysis.json) for per-task outcomes, paired wins/losses and task-bootstrap intervals; [manifest.json](manifest.json) for checkpoint and source hashes; [report.json](report.json) for timing and memory.

These are development measurements from one training seed. They do not establish the paper's overall efficacy. The state/shape audit does not judge goal-image semantics. Initial predicate satisfaction is reported separately.

**Short runtime smoke only. These truncated episodes do not measure task performance.**
