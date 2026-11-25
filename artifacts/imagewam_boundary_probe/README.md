# ImageWAM at LIBERO-10 subtask boundaries, with our subtask wording (2026-10-29)

Follow-up to the [LIBERO-Goal language test](../imagewam_goal_language_probe/README.md), where ImageWAM followed familiar LIBERO-Goal instructions from start frames. This is the case closest to π0.7's use: **our** subtask instructions, at the moment each LIBERO-10 subtask starts.

## Setup

- **Data:** `libero_10_subtask` (our training data; 256 px lerobot renders, a different rendering source from ImageWAM's training images). One episode from each two-step task (tasks 0–8; task 9 has one segment). Episodes 42, 82, 125, 162, 215, 252, 303, 350, 408 (random, seed 0).
- **Two boundaries per task:**
  - **B0 = frame 0.** Subtask A is about to start. In most tasks subtask B could be done first instead (turn on the stove *or* place the moka pot; soup *or* cream cheese into the basket).
  - **B1 = first frame of subtask B**, with A done. The gripper is often still at the object it just placed (tasks 1, 2, 8), so this isn't a moment of rest.
- **Prompts:** subtask A text, subtask B text (wording from the [dataset.md](../../dataset.md) table), full task text, empty.
- **Generation:** for each boundary and prompt, chains of 5 calls (80 steps) for seeds 0 and 1, with the state held at the boundary. The first call is the single-step (16-step) prediction. 720 images at 1.40 s each.
- **Inputs:** [inputs_preview.png](inputs_preview.png).

## Result

**This probe does not establish reliable subtask selection from our instructions. The inspected examples largely follow the usual scene-specific continuation.**

| Measure | 16 steps | 80 steps |
|---|---|---|
| Right subtask's picture closer to the real continuation than the wrong subtask's | 0.64 (B0: 0.50, B1: 0.78) | 0.50 (B0: 0.50, B1: 0.50) |
| Difference: right vs wrong instruction (same seed) | 0.0021 | 0.0148 |
| Difference: seed 0 vs seed 1 (same instruction) | 0.0081 | 0.0168 |

(Mean squared difference at 56 px, both cameras. Chance for the first row is 0.5.)

- **B0 (either subtask possible):**
  - Task 0: "turn on the stove", "put the moka pot on the stove" and empty text give the same picture. The arm goes to the knob and the burner turns red.
  - Task 4: the soup and cream-cheese instructions both pick up the same item.
  - The other tasks are similar. The instruction changes the picture less than the random seed does.
- **B1 (next subtask vs the one already done):**
  - The right next instruction, or the full task text, moves the arm on to the next step.
  - The already-done instruction often makes the model **stall**: task 0, "turn on the stove" when it's already on, leaves the arm hovering at the knob, just like empty text (distance to reality 0.096 vs 0.027 for the right instruction).
  - So the text works as a go/stop switch, not as a choice between steps.
- **The heading check** (does the B0 prediction look more like "A in progress" or "B in progress") said "A" for every prompt. Treat it as supporting evidence only: its reference for A shares more background with the start frame than its reference for B does.

## Why this differs from LIBERO-Goal

A plausible explanation is that ImageWAM learned to use language more strongly where its training required it. In LIBERO-Goal, one scene appears with 10 instructions, so the image can't tell it what to do and the words must. In LIBERO-10, each scene appears with one task and always in the same subtask order, so the scene alone predicts the next step and the words were never needed to choose. The rendering source and wording also differ from the Goal test, but the pattern (same output under contradictory instructions, stalling under an already-done instruction) points to scene-learned habits.

## What it means for the π0.7 toy

- **Out of the box on LIBERO-10, ImageWAM gives the habitual next step, not the instructed one.** Because LIBERO-10 demos always use that order, its goal pictures will usually still be the "right" next step for our tasks. But it fails the steering requirement: a different subtask instruction won't produce a different goal.
- **Fixed-order fine-tuning does not establish steering.** It could improve subtask vocabulary or endpoint prediction, but scene cues remain sufficient to explain most demonstrations. Reliable redirection needs instruction-dependent outcome examples and a matching test: LIBERO-Goal, or feasible reordered LIBERO-10 rollouts. Avoid claiming that fine-tuning is proven incapable of helping; it has not been tested. The audited task-8 data also contains one reversed-order episode, so the order is overwhelmingly fixed rather than universally fixed.
- **Option:** test the steering part of the experiment on LIBERO-Goal, where ImageWAM already follows instructions, as [dataset.md](../../dataset.md) already suggests for instruction-sensitive choices.

## Limits

One episode per task, 2 seeds, and a pixel-distance score near chance. The visual check of tasks 0, 4, 5 and task 0 B1 is the main evidence. Simulator rollouts were not used.

## Files

- `run_probe.py` (everything), `inputs_preview.png`.
- `results/B0_main.png`, `results/B1_main.png` (+ `_wrist`). Rows per task: ALIGNED / OTHER / EMPTY. Columns: start | seeds 0–1 at 16 steps | chains at 80 steps | real 16 and 80 steps later.
- `results/report.json`, `results/predictions/`, `run.log`.

## Review and plan update (2026-10-29)

Recomputed the paired scores from saved predictions and freshly decoded references: 23/36 aligned wins at 16 transitions and 18/36 at the nominal 80-transition horizon. Instruction/seed MSEs reproduced (0.00205/0.00813 and 0.01481/0.01679). No B0 reference crossed into B1 for these selected cases. The task-0 B0/B1 full-size examples support the stated lack of initial redirection and subsequent prompt-dependent stalling.

The 50% figure is a paired pixel-distance win rate against the observed continuation, not an instruction-following success rate or proof of zero language effect. No valid alternative execution is available as ground truth. B1's already-completed instruction is also not generally a choice between two equally feasible new actions. The reported 2 clear / 1 slight cases are qualitative judgments, not thresholded simulator outcomes. Constant state during image chaining and rendering/wording differences remain additional limitations.

The [revised plan](../../reconstruction.md) separates LIBERO-10 visual-guidance benefit from LIBERO-Goal familiar-instruction control, keeps initial A/B text inputs matched, and reserves reordered-task and metadata claims for separate tests.
