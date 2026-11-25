# Next tasks

**Current resume point: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), 2026-11-04.**
The implementation goal is active. Entries below preserve the history of
diagnostics; their older "next" instructions are not the current queue.

Updated 2026-10-30 after the baseline discussion and Section V implementation. Main plan and resume point: [reconstruction.md](reconstruction.md). Measurements and decision rules: [run_scorecard.md](run_scorecard.md).

## Three-arm rerun (A / B-near / B-end) from `smolvla_libero_90` (2026-10-30, analysed 2026-11-03): no measurable goal benefit

Same recipe as below (alt layout, lr 1e-4, 2,000 updates, batch 8, seed 20261029). A and B-near reproduce the earlier run exactly. **B-end** trains on the current subtask's last frame (π0.7's subgoal form). Checkpoints: `/media/skr/storage/checkpoints/pi07_goal_ab/l90v2_{A,Bnear,Bend}/`. Reports with per-observation errors: `artifacts/goal_fix/l90v2_*/report.json`. Held-out set: 100 observations from 43 episodes. The wrong-task check now always pairs different tasks (the old shift-10 pairing also had 0 same-task pairs).

First-8 MSE at 2000 (start 0.551; A = 0.421):

| Held-out goal | B-near | B-end |
|---|---|---|
| near real (8–16 ahead) | **0.410** | 0.427 |
| end real (subtask end) | 0.425 | 0.430 |
| original (0–40 / end) | 0.418 | 0.434 |
| near wrong-task | 0.423 | 0.432 |
| end wrong-task | 0.425 | 0.439 |
| copied current | 0.404 | 0.424 |
| dropped | 0.452 | 0.470 |

Paired differences on each B arm's own goal type, 95% CI by resampling episodes:

| | own goal − A | own − copied current | own − wrong task |
|---|---|---|---|
| B-near | −0.010 [−0.022, +0.002] | +0.006 [−0.007, +0.020] | −0.012 [−0.029, +0.003] |
| B-end | +0.010 [−0.009, +0.033] | +0.006 [−0.006, +0.020] | −0.008 [−0.025, +0.006] |

Every interval includes zero. After 2,000 updates neither goal type gives a measurable benefit over A, over a copy of the current frame, or over a wrong-task goal. B-end with endpoint goals isn't better than B-near. Per plan step 3, **no simulator run yet**. Options: train longer from the saved checkpoints (needs `--resume`), more seeds, or rethink how goals enter the model.

## A/B from `smolvla_libero_90` (2026-10-30): B ties A, and goal *content* isn't used yet

`train_goal_fix.py --layout alt --base /media/skr/storage/models/smolvla_libero_90_pi07 --lr 1e-4 --steps 2000 --eval-every 500`, arms A and B run in parallel (about 22 min each, 4.7 GB each). B has a goal frame 8–16 steps ahead in **100%** of examples. Reports: `artifacts/goal_fix/l90_{A,B}/` (report.json rebuilt from the logs after a Path-serialization crash at the very end, since fixed; curve_*.json). Weights were not saved.

Held-out first-8 / 50-step action MSE (start 0.551 / 0.746):

| Update | A | B real goal | B goal off |
|---|---|---|---|
| 500 | 0.597 / 0.635 | 0.650 / 0.670 | 0.625 / 0.653 |
| 1000 | 0.485 / 0.579 | 0.509 / 0.563 | 0.488 / 0.588 |
| 1500 | 0.492 / 0.542 | 0.486 / 0.559 | 0.502 / 0.563 |
| 2000 | **0.421 / 0.527** | **0.417 / 0.519** | 0.452 / 0.568 |

At 2000, B with copied current images scored 0.404 / 0.517 and B with wrong-task goals 0.426 / 0.531. Training loss over the last 250 updates was A 0.452 and B 0.459.

- **B ties A** (0.417 vs 0.421). It doesn't learn faster, and its training loss stays slightly higher throughout.
- **B depends on having *a* goal image** (goal off is worse, 0.452), but copying the current frame works as well as the real future frame (0.404), and wrong-task goals are only slightly worse (0.426). So B uses the goal slot as an extra view of the scene, not as information about the future.
- **Caveat: held-out goals are farther ahead than training goals.** Validation uses `goal_training_1000`'s 0–40-frame or segment-endpoint offsets; training used 8–16. The real-goal score may be understated.
- **Next options:** (1) re-run with a held-out mode that uses 8–16-frame goals and save the weights; (2) train longer (the paper's effect may need more than 2,000 updates at batch 8); (3) more seeds, since differences of 0.01–0.02 on 100 observations are within noise.

## Start checkpoint for the next A/B: `smolvla_libero_90` (2026-10-30)

`lerobot/smolvla_base` is for the SO100 arm (6-D state/action, 3 cameras, SO100-only normalization), so it would need adapting. Downloaded instead `Kesvill/smolvla_libero_90` (`/media/skr/storage/models/smolvla_libero_90`, community upload with no model card). It was trained 100k steps at batch 8 on a LIBERO-90 episode subset of `nvidia/LIBERO_LeRobot_v3` and has never seen LIBERO-10 demonstrations. It has the same 500 weight tensors and shapes as `smolvla_libero`, 8-D state and 7-D action with the same state convention, LIBERO normalization stats, and cameras `image`/`image2` (no rename needed). Its config has inactive `compile_model`/`compile_mode` fields that our loader rejects, so [libero90_check.py](libero90_check.py) loads a cleaned copy.

Untouched, on the same 100 held-out observations ([report](artifacts/libero90_check/report.json)): first-8 / 50-step MSE **0.551 / 0.746** as rendered, 0.589 / 0.762 with images rotated 180° (keep as rendered). `smolvla_libero` scores 0.123. Both are measured in each checkpoint's own normalized units, but action stds agree within about ±25% per dimension. So 0.123 reflects having trained on these demonstrations, and the LIBERO-90 checkpoint leaves room for goal images to help. Per-task error ranges from 0.17 (task 9) to 0.91 (task 0).

**Next:** the A/B capability test from `smolvla_libero_90` (isolated goal layout, 100% near-future goals for B, matched A), comparing learning curves and held-out error. `train_goal_fix.py` needs a `--base` option and the same cleaned-config loading.

## Tasks 1–2 results (2026-10-30): no fix makes goals help; next test starts from `smolvla_base`

[train_goal_fix.py](train_goal_fix.py); results in `artifacts/goal_fix/{cap,cap_fix,cap_alt}/report.json`. Same observation schedule and seed as `goal_training_1000`, lr 1e-5, 1,000 updates, goals in **100%** of training examples at 8–16 frames ahead. Held-out first-8 action MSE (the validation goals use the original 0–40-frame / endpoint offsets):

| Run | Real goal | Goal off | Copy current | Wrong task | Train loss (last 250) |
|---|---|---|---|---|---|
| B, Codex layout (`cap`) | 0.323 | 0.227 | 0.281 | 0.364 | 0.417 |
| B + goal marker + last 4 VLM layers unfrozen (`cap_fix`) | 0.322 | 0.224 | 0.281 | 0.362 | 0.416 |
| A, same unfreezing (`cap_fix`) | – | 0.227 | – | – | 0.390 |
| B, isolated goal layout: images + text one block, goals after text (`cap_alt`) | 0.309 | 0.216 | 0.260 | 0.339 | 0.401 |
| A, isolated layout (`cap_alt`) | – | 0.228 | – | – | 0.390 |

- Ruled out: the marker, unfreezing 4 layers, text contamination by goal tokens (the isolated layout helps a little, but goals still hurt), and misaligned goal frames (10/10 checked goals come from the same episode at the scheduled offset).
- Even in training, B with a near-future goal in every example has a *higher* loss than A, the opposite of the paper's "subgoals make training significantly faster".
- **Leading explanation:** `smolvla_libero` was already trained on these LIBERO-10 demonstrations (untouched held-out error 0.123), so the current image already gives the answer and goal tokens act as a distraction. The paper's effect needs a policy that doesn't already know the demonstrations.
- **Next test:** start A and B from `lerobot/smolvla_base` (not LIBERO-trained; about 0.9 GB download), train both on `libero_10_subtask` for a few thousand updates, and compare learning curves and held-out action error. If the goal pathway works, B should learn clearly faster. Use the isolated layout. About 45–60 min of GPU.

## 0. Attention-layout check: DONE 2026-10-30, layout is not the problem

[attention_check.py](attention_check.py), [report](artifacts/attention_check/report.json). Untrained base checkpoint, the 100 held-out observations of `goal_training_1000`, goals masked, same noise seeds as `evaluate_goal_pilot.py`. First-8 / 50-step action MSE:

| Layout | MSE first 8 | MSE 50 |
|---|---|---|
| Original SmolVLA | 0.1229 | 0.2035 |
| Codex goal layout | 0.1227 | 0.2040 |
| Alternative (images + text one block, goals after) | 0.1228 | 0.2037 |

Codex's layout does not degrade the pretrained policy on average. Keep it.

**New finding: fine-tuning made the policy worse offline.** The untrained base scores **0.123**; arm A after 1,000 updates scored **0.289** on the same observations with the same method. Normalization statistics for action and state are identical in both checkpoints, and the saved configs differ only in trainable modules and `n_action_steps`. The time/noise sampling matches SmolVLA's Beta(1.5, 1) convention. Meanwhile A's held-out *flow* loss fell (0.64 → 0.35), so flow loss and sampled-action error disagree. Likely contributors, not yet tested:
- The base was trained on `lerobot/libero`, which contains these LIBERO-10 demonstrations, so its low error on the "held-out" episodes may be memorization that fine-tuning partly overwrites.
- Learning rate 1e-4 may be high for an already converged checkpoint.

**Low-learning-rate check: DONE 2026-10-30** ([run](artifacts/goal_training_1000_lr1e-5/offline_action_diagnostics/report.json); same seed and example schedule, lr 1e-5 instead of 1e-4). First-8 / 50-step action MSE:

| Condition | Base (untrained) | lr 1e-4 | lr 1e-5 |
|---|---|---|---|
| A / goals dropped | 0.123 / 0.204 | 0.289 / 0.331 | 0.227 / 0.262 |
| B real future goals | – | 0.340 / 0.357 | 0.363 / 0.381 |
| B copied current | – | 0.324 / 0.356 | 0.326 / 0.375 |
| B mismatched task | – | 0.355 / 0.372 | 0.407 / 0.436 |

The lower learning rate reduces the damage but doesn't remove it: A is still about 1.8× the base's error. Goals still hurt B; real goals now beat mismatched ones by a wider margin (0.363 vs 0.407), so B reads some goal content. The base may simply have memorized these demonstrations (it was trained on `lerobot/libero`), which would make offline MSE an unfair yardstick. **Next: closed-loop base vs A (lr 1e-5) on all 10 tasks × development starts 0/1**, to see whether fine-tuning hurts actual task success.

**Original next-diagnostic note:** evaluate sampled-action MSE of A's saved checkpoints at several update counts, and a short low-learning-rate A run (e.g. 1e-5). If A stays near the base's error, use that setting for both arms. Closed-loop A must not fall below the base checkpoint, or the A/B compares two degraded policies.

### Original task 0 description

**Why.** [goal_policy.py](goal_policy.py) puts the first goal token in a new attention block (`blocks[obs_tokens] = 1`), so current-camera tokens can no longer attend to the task text. Original SmolVLA keeps images and language in one bidirectional block. The layout is used for **both** A and B, while the VLM is frozen. It may be degrading the pretrained features in both arms, which would fit A not beating the unmodified checkpoint (0/6 vs 1/6 on the development starts) and B struggling with goals.

**Check.** On the same 100 held-out observations, measure the unmodified checkpoint's action error (first-8 and 50-step MSE) under (i) the original SmolVLA prefix layout and (ii) the goal layout with goals masked. No training.

**Read-out.** If (ii) is clearly worse, use a layout that keeps current images + text in one block as originally trained, with goal images in a later block that can see images and text but isn't seen by them (consistent with π0.7's attention rules). Use that layout in task 1.

## 1. Goal-pathway capability test (after task 0; the closed-loop development run has finished)

**Why.** In the [1,000-update matched run](artifacts/goal_training_1000/README.md), B performed *worse* with real future goals than with goals dropped. First-8 action MSE on the 100 held-out observations:

| Condition | MSE |
|---|---|
| A, no goals | 0.289 |
| B, goals dropped | 0.288 |
| B, real future goals | 0.340 |
| B, copied current images | 0.324 |
| B, mismatched-task goals | 0.355 |

- Real and wrong-task goals produce nearby error scores. This alone does not establish how much B uses goal content.
- The [validation curve](artifacts/goal_training_1000/validation_curve.png) shows the goal/no-goal gap narrowing; it does not distinguish learning to use goals from learning to suppress unhelpful inputs.
- Real near-future frames constrain possible motion but do not uniquely specify actions. Improvement would support useful goal conditioning; a null result needs further diagnosis.
- This test asks only whether the pathway *can* use goals, separately from π0.7's 25% goal-presence recipe.

**Setup.**
- Same base checkpoint, trainable modules, optimizer, batch size 8, 1,000 updates and observation schedule as `goal_training_1000`; reuse its A as the reference.
- **B-cap:** goals present in **100%** of examples. Goal = the real frame **8–16 steps ahead** in the same episode, clipped at the episode end.
- Offline evaluation with `evaluate_goal_pilot.py` conditions (real, dropped, copied-current, mismatched-task) on the same 100 held-out observations.
- Write to a new output directory, e.g. `artifacts/goal_capability_test/`. Don't overwrite existing runs.

**Read-out** (working criteria for this experiment, not paper thresholds):
- **Pathway works:** B-cap's real-goal MSE is clearly below A's (for example ≥20% lower) and clearly below its copied-current and mismatched-task scores. The problem is then training scale or recipe; continue with a longer matched budget.
- **Investigate:** no clear drop. Use task 2 to diagnose representation or optimization; this does not prove that the pathway cannot use goals.

**Cost.** About 10 min training + 5 min evaluation; about 5 GB GPU.

## 2. Goal-pathway fixes (if task 1 shows the pathway is blocked)

Change one thing at a time and re-check each with the task 1 capability test:

1. **Goal marker.** Let the model tell goal images from current camera views. Currently nothing marks them, and the frozen VLM sees them as two more cameras. Options: a text tag (e.g. "goal:") before the goal tokens, or a small learned embedding added to the goal tokens.
2. **Partly unfreeze the VLM** (LoRA or the last few layers) so goal images can get goal-specific features. Memory headroom: 4.7 GB peak of 24 GB.
3. **Longer training budget,** matched for A and B.

Then go back to the 25% goal-presence recipe and repeat the matched A/B comparison.

## Context

- The closed-loop development comparison is complete: A, B dropped and B generated each scored 0/6, with zero runtime errors. Treat it as integration evidence, not proof of equal efficacy. The Section V one-update CPU smoke is also complete; no production Section V training has run.
