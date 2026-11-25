# SmolVLA pretrained reference pilot

Completed 2026-10-29. No training, goal images or ImageWAM calls. This establishes a functioning reference pipeline before the matched A/B experiment.

## Fixed setup

- Policy: `/media/skr/storage/models/smolvla_libero`, strict checkpoint loading; SHA-256 saved in the [manifest](manifest.json).
- Two suites: `libero_10`, `libero_goal`; simulator task IDs **0, 4, 8**; initial-state IDs **0, 1**. All 12 jobs were written to the manifest before rollouts.
- 50-action output chunks, first **8** executed; 10 denoising steps. Two RGB cameras, 8-D state, full task instruction; saved checkpoint processors.
- 10 settling steps; simulator **20 Hz**. Limits: 520 policy steps for LIBERO-10, 300 for LIBERO-Goal. Simulation pauses for inference.
- Actions clipped to [-1,1], with clipping recorded. Videos contain both cameras sampled every 8 simulator steps, encoded at 2.5 fps.
- These are **development starts**. Exclude these task/start combinations from the final paired evaluation and do not use this pilot to claim unseen-model generalization.
- Simulator task IDs differ from dataset task IDs. Match tasks by their full language/name, not by equating numeric IDs across sources.

## Quantitative results

| Measure | LIBERO-10 | LIBERO-Goal |
| --- | --- | --- |
| Successes | **1/6 (16.7%)** | **6/6 (100%)** |
| Descriptive Wilson 95% interval | 3.0–56.4% | 61.0–100% |
| Timeouts | 5/6 | 0/6 |
| Runtime exceptions | 0 | 0 |
| Mean steps among successful episodes | 229 | 94 |
| Chunk inference median / p95 | 0.141 / 0.144 s | 0.141 / 0.146 s |
| Rollout wall time (includes environment setup and video) | 102.8 s | 43.5 s |

Only three predetermined tasks and two starts per task were used. The intervals are descriptive small-sample summaries, not calibrated claims about the full heterogeneous benchmark or variation across training seeds. Do not interpret the 100% pilot result as guaranteed performance or available A/B headroom across all LIBERO-Goal tasks.

Task results: LIBERO-10 soup+tomato sauce **0/2**, two mugs onto plates **1/2**, both moka pots **0/2**. LIBERO-Goal middle drawer, bowl onto cabinet, and bowl onto plate were each **2/2**.

Across both suites: **3,393 policy steps**, **23,751/23,751 finite action components**, peak allocated GPU memory **1.271 GB**, peak reserved **1.321 GB**. These are PyTorch figures for SmolVLA alone; they exclude an ImageWAM process and are not total-device or training memory.

Clipping affected **7.76%** of action components, overwhelmingly gripper values: **54.2%** of gripper commands slightly exceeded [-1,1] (maximum magnitude 1.0973). Of the six motion components, only z translation exceeded bounds: **5/3,393** commands, maximum magnitude 1.0207. Preserve this clipping convention across A and B. [Action audit](action_audit.json).

## Checkpoint decision

**Continue to goal-conditioning implementation.** The runtime is functional and requested-goal successes occur. LIBERO-10 performance is weak in this small pilot, so it is a reference to improve and compare against, not a validated high-performance baseline. Review failure videos when selecting development diagnostics. LIBERO-Goal needs the broader planned evaluation because these three tasks leave no measured success headroom in the pilot.

The inspected contact sheets show task-directed motion and the requested changes in successful Goal episodes; LIBERO-10 failures show incomplete manipulation rather than an obvious globally inverted camera stream. This is a qualitative spot-check, not a complete physical-error analysis.

## Artifacts

- [Manifest](manifest.json): exact jobs, seeds, checkpoint hash, source revisions and settings.
- [Report](report.json): per-episode outcomes, timings, per-task and per-suite summaries.
- [LIBERO-10 contact sheet](libero_10_contact_sheet.png), [LIBERO-Goal contact sheet](libero_goal_contact_sheet.png).
- Each `.mp4` shows main | wrist views. Each `*_trajectory.npz` stores raw policy actions before clipping, observed states and chunk inference times.
- [Runner](../lerobot_libero_setup/run_reference_pilot.py), [log](../lerobot_libero_setup/reference_pilot.log).
