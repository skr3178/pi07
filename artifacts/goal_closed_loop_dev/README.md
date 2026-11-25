# Live generated-goal development comparison

This tests the complete current-observation → ImageWAM → goal-conditioned
SmolVLA → LIBERO loop. It uses the final checkpoints from the
[1,000-update matched run](../goal_training_1000/README.md).

**Completed 2026-10-29. Decision: investigate policy readiness and goal
conditioning before the final benchmark.** The live integration works, but
this development run shows no task-success benefit. All jobs have finished.

## Results

| Condition | Successes | Timeouts | Runtime errors | Rollout wall time |
| --- | ---: | ---: | ---: | ---: |
| A, goals dropped | 0/6 | 6/6 | 0 | 128.5 s |
| B, goals dropped | 0/6 | 6/6 | 0 | 128.7 s |
| B, ImageWAM goals | 0/6 | 6/6 | 0 | 416.2 s |

All 18 episodes ran to the 520-action limit. All **65,520/65,520** action
components were finite. Clipping affected 7.40% / 7.80% / 7.77% of components,
almost entirely the gripper; there was only one clipped motion component
across the entire comparison (A's z action reached 1.0068).

The [audit](audit.json) checks **198/198** generated image pairs and their
current inputs. Generation state matches the stored trajectory state at the
same step exactly (maximum error 0.0). The initial 8-D robot states also match
across conditions exactly. These are robot-state checks, not a new audit of
every simulator object state; the earlier reset-state audit validates the
wrapper's requested-state restoration.

Generation latency was **1.423 s median / 1.430 s p95** per pair. ImageWAM
peak allocated/reserved memory after unloading Qwen was **9.505 / 9.947 GB**;
the policy peak was **1.282 GB** in its separate process. The sum of separate
process peaks is not a measured device-wide peak. Worker startup took 16.4 s
and peaked at 17.607 GB before the policy loaded. Rollout wall times exclude
checkpoint/worker startup. Generated guidance cost about 3.24× the no-goal
rollout time in this simulator setup.

All six paired outcomes are ties. With only two starts on each of three tasks,
this cannot establish equal performance or rank the conditions. The audit's
illustrative binomial Wilson interval for 0/6 is approximately 0–39%; it is
not a suite-wide uncertainty estimate. The earlier unmodified checkpoint
scored 1/6 on these starts; one changed outcome is insufficient to establish
a robust regression.

## Next diagnostic

The existing [todo.md](../../todo.md) queues a **goal-pathway capability test
first**. Preserve that priority: train B-cap from the same base with 100% real
goals 8–16 frames ahead, using the saved observation/action/noise schedule and
matched 1,000-update budget. Reuse A as the reference and evaluate real,
dropped, copied-current and mismatched goals on the same held-out observations.
This is a diagnostic departure from the 25% recipe, not a replacement main A/B.
It has not run. A null result would call for investigation, not prove that the
pathway cannot use goals; action MSE does not uniquely identify the cause.

Follow-up controls, after that queued diagnostic:

1. If adaptation is weak, isolate the changed attention layout from training:
   use a matched no-goal control with the original observation/text attention,
   identical trajectories/noise/budget and the same corrected padding loss.
   This is a proposed control, not a completed run.
2. Recheck B with real future, copied-current and dropped goals on the fixed
   offline set while diagnosing the goal representation/training budget. Real
   futures already worsen action MSE (0.3403 versus 0.2882 dropped), so swapping
   generators alone does not address the observed offline gap. This is a
   diagnostic lead, not proof of a single cause.
3. Before the final benchmark, establish the no-goal baseline on development
   starts 0/1 across **all ten tasks**, keeping the final starts unused. Compare
   the unmodified checkpoint with adapted A and inspect failure types.

The final 100-start comparison is **not run**. No claim about the paper's
general method follows from this small, short-training development result.

## Fixed protocol

Three conditions use identical simulator tasks 0/4/8, initial-state IDs 0/1,
policy random seeds, 520-step limits and 10 settling steps:

- A: goal slots masked.
- B dropped: same trained B, goal slots masked.
- B generated: same trained B, goals from frozen ImageWAM.

All conditions generate 50 policy actions and execute 8 before replanning.
ImageWAM predicts one main/wrist pair every 16 simulator actions, using only
the current images, measured 8-D state and full task instruction. Its ten
denoising steps predict a 16-retained-transition continuation, not a verified
semantic subtask endpoint. There is no chain and no future demonstration state.

The worker persists in the separate ImageWAM environment, caches the three
task-text embeddings, and unloads Qwen before rollout. Camera orientation is
already corrected by LeRobot; the worker resizes each 256px view to 224px and
concatenates main then wrist. Its output is split and resized to 256px for the
policy. State gets ImageWAM min/max normalization, independently of SmolVLA's
saved mean/std processor.

Simulation pauses during generation and inference. Reported elapsed time is
compute-paced evaluation time; this does not demonstrate real-time control.

## Evidence and scope

- [Manifest](manifest.json) pins task names, starts, seeds, model/config/stats
  hashes, source revisions and generation settings before evaluation.
- [Report](report.json) records live status, outcomes, per-refresh latency,
  finite/clipped actions, policy memory and worker startup/memory.
- MP4s contain main and wrist cameras. Every generated goal PNG has a matching
  `_current.npz` with the actual current images and measured state; trajectories
  preserve the policy actions and observed states.
- [Source snapshot](source/) preserves the runtime adapters.
- [A rollout sample](A_development_frames.png) shows four recorded times for
  each of the three tasks. Visible partial manipulation is not counted as
  success; the LIBERO success predicate is authoritative.
- [Live goal examples](generated_goal_examples.png) show current and generated
  main/wrist views at three times in task 0. These were inspected for orientation
  and gross scene preservation; no semantic goal-quality score is inferred.

These six starts are for development only. Their outcomes are too few to
support a reliable efficacy claim. The [reserved final 100 starts](../libero10_evaluation_design/README.md)
are not used here. No real future goal is supplied in a live rollout.
