# Section V prompt implementation

Implemented and checked **2026-10-30** against the local [π0.7 paper](../../pi_07.pdf),
Section V-A–E. This adds the missing input/data/training code. It does **not**
establish that a full-prompt policy performs better. The original baseline and
the previous A/B implementations/checkpoints remain unchanged.

## What is implemented

| Subsection | Implementation | Remaining experimental limit |
| --- | --- | --- |
| V-A: subtask instructions | Full task plus explicit `Subtask:` text, audited ID mapping, frame-aligned offline labels, provenance, and a runtime request API | A live human/controller must supply the current subtask. Demo boundary times cannot be replayed as live observations. ImageWAM must receive the corresponding intended subtask when this is connected online. |
| V-B: subgoal images | Reuses the tested main/wrist goal slots and masks, episode-safe real-future pairs, and the existing ImageWAM worker | Generated-image benefit remains unproven; current ImageWAM predicts a fixed-horizon continuation rather than a validated semantic endpoint. |
| V-C: episode metadata | Typed speed, quality and mistake fields; provenance/confidence; reviewed annotation loader; missingness kept distinct from dropout; requested metadata at inference | Demo lengths are available, but actual quality and mistake annotations are absent. Mixed-quality learning still needs matched-task rollouts and reviewed labels. |
| V-D: control mode | Explicit `Control Mode: ee.` in every profile; unsupported joint mode is rejected | Only the existing end-effector controller is implemented. A constant tag does not demonstrate mode switching. |
| V-E: dropout | Goals present 25%; subtask dropped 30% **when a goal is present**; whole metadata dropped 15%, plus each field 5%; control mode never dropped | Input dropout is implemented and checked. Performance with missing prompts still needs trained-policy evaluation. |

The complete text format is `Task: … Subtask: … Speed: … Quality: … Mistake: …
Control Mode: ee.` Missing/disabled/dropped fields are omitted. Their values and
provenance remain in separate audit records, outside the policy inputs.

## Measured checks

| Check | Result |
| --- | --- |
| Synthetic dropout trials | 10,000; all measured rates within the declared three-sigma range |
| Goal presence | 24.37% (target 25%) |
| Subtask dropout conditional on a goal | 29.71% (target 30%) |
| Whole-metadata dropout | 15.30% (target 15%) |
| Individual field presence after both dropouts | Speed 80.31%, quality 80.68%, mistake 80.71% (target 80.75%) |
| Control-mode presence | 10,000/10,000 |
| Real-data sampled pairs | 10,000; zero cross-episode goals |
| Unknown real-data quality/mistake labels | Both remain unavailable in all 10,000 audited samples |
| Subtask mapping | All 16 IDs covered; task 3's first-frame artifact repaired in memory in 50 episodes |
| Decoded/preprocessed real samples | 10/10, spanning all ten tasks |
| Full-field text examples | 20/20 exceed the old 48-token limit; maximum 65 tokens |
| New text budget | 128 tokens; overlength requests rejected, silent truncation disabled |
| Saved processor roundtrip | Token IDs exactly equal |
| Real model forward/backward | Passed on CPU; 155/155 gradient tensors finite; no optimizer updates in this check |
| Separate training-entry smoke | One CPU optimizer update completed; frozen backbone unchanged; checkpoint save/reload exactly equal |

The one-update smoke uses one training and one validation example. It is
strictly an integration check, not a new baseline or a useful trained policy.
Its checkpoint must not be used in an efficacy comparison.

[Detailed gates](report.json), [prompt examples](prompt_examples.json),
[training-entry smoke](../section_v_training_smoke/report.json), and
[matched preparation audit](preflight_comparison.json) are saved locally.

## Prepared experiment profiles

`train_section_v.py` supports `task`, `subtask`, `goal`, `subtask_goal` and `full`.
Every profile uses the same 128-token budget, control tag, fixed four image
slots, base weights, saved observation/action schedule and optimizer recipe.

The [task-only preparation](../section_v_task_preflight/manifest.json) and
[full preparation](../section_v_full_preflight/manifest.json) use the same
8,000 observations and future indices. Full has 2,058/8,000 visible goals
(25.725%); task-only has zero. Both have 8,000 known subtask/length annotations
and **zero known quality/mistake annotations**. No production training has
been launched for these profiles.

**A fresh task-only control is needed for these new profiles.** The previous
A checkpoint used the original text format and 48-token budget. It remains
a useful earlier baseline, but cannot isolate changes introduced by this new
prompt format. The downloaded unmodified LIBERO checkpoint remains the reference.

For a text × goal factorial, set `--subtask-drop-given-goal 0` in the compared
text arms so adding goals does not also reduce text availability. Then use the
paper's conditional 30% dropout in a separately declared full-recipe experiment.

## Data and annotation limits

Speed is recorded episode length, **not an inferred success/quality score**.
The local default uses 50-step bins because LIBERO episodes are short; the
paper uses 500-step bins (`--speed-bin-steps 500`). Rounding is half-up, with a
minimum of one bin for short positive episodes. These are retained dataset
frames; their mapping to physical time is not verified. Do not silently combine
these counts with simulator-step durations from a different sampling convention.

Quality and mistake annotations can be loaded from JSONL via `--annotations`.
A valid empty template is:

```json
{"episode_index": 0, "quality": {"value": null, "source": "unavailable", "confidence": null}, "mistakes": []}
```

Reviewed quality values must be integers 1–5. Mistake intervals have
`start_frame`, `end_frame_exclusive` and a `label` object containing a boolean
`value`, a nonempty `source`, and optional `confidence` (unknown or 0–1).
Intervals must be within the episode and nonoverlapping. Uncovered frames stay
unknown. Duplicate episodes, malformed fields and invalid ranges are rejected.
A terminal success bit is deliberately not accepted as a substitute for a
five-level quality score or a segment-level mistake label.

The synthetic labels in the gate check are explicitly test fixtures; they
never become training annotations. The current `full` preparation can test
duration conditioning, but cannot test the paper's mixed-quality claim yet.

## Code and commands

- [section_v_prompt.py](../../section_v_prompt.py): field schema, deterministic
  dropout, runtime requests and guarded tokenizer/processor.
- [section_v_data.py](../../section_v_data.py): subtask map, annotation ingestion,
  dataset adapter and separate audit records.
- [train_section_v.py](../../train_section_v.py): single-profile training,
  source/data manifests, finite-loss checks and exact checkpoint roundtrip.
- [check_section_v.py](../../check_section_v.py): measured implementation gates.

Prepare inputs without training, using a **fresh output directory**:

```bash
/media/skr/storage/envs/pi07-lerobot/bin/python pi07_code/train_section_v.py \
  --profile full --prepare-only --output pi07_code/artifacts/NEW_PREFLIGHT
```

For a subsequent approved experiment, omitting `--prepare-only` runs the
declared training budget (default 1,000 updates, batch size 8) on CUDA. The
runner refuses to overwrite an existing manifest. Load resulting checkpoints
with `GoalSmolVLAPolicy` and `make_section_v_processors`; use the saved prompt
options from `section_v_interface.json`.

The next project priority is **broader scoring of the existing reference and A**,
followed by the goal-use diagnostic and controlled prompt ablations. This input
implementation does not replace those gates. Metadata CFG belongs to Section
VII and is not implemented here. Joint control and a learned subtask planner
also remain outside the current implementation.
