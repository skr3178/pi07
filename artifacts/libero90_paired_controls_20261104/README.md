# LIBERO-90 live development evaluation

Completed 2026-11-04.

Same task/start pairs, seeds, 50-action predictions with eight executed, 520-step limits.
Generated goals use one ImageWAM call from current main/wrist images and state every 16 simulator actions.
Simulation pauses during inference. No demonstration future is supplied online.

| Condition | Successes | Mean maximum predicate fraction | Mean gain over initial state |
| --- | --- | --- | --- |
| reference | 0/20 | 0.133 | 0.100 |
| A | 2/20 | 0.283 | 0.250 |
| Bnear_off | 0/20 | 0.133 | 0.100 |
| Bnear_generated | 4/20 | 0.308 | 0.275 |
| Bnear_copy | 4/20 | 0.333 | 0.300 |

Verified 100 episodes and finite actions; 80 cross-condition full simulator reset comparisons. Audited 599 generated pairs and 593 copied-current pairs; maximum generator/trajectory state difference 0.0.

See [analysis.json](analysis.json) for per-task outcomes, paired wins/losses and task-bootstrap intervals; [manifest.json](manifest.json) for checkpoint and source hashes; [report.json](report.json) for timing and memory.

These are development measurements from one training seed. They do not establish the paper's overall efficacy. The state/shape audit does not judge goal-image semantics. Initial predicate satisfaction is reported separately.

GPU diagnostics overlapped part of this evaluation; see [timing context](timing_context.json). Overlapping wall/inference times are not an isolated throughput comparison.
