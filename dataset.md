# Dataset options

As of 2026-10-28. Companion to [reconstruction.md](reconstruction.md) and [model.md](model.md).

**Selection objective: Section V, “Diversifying the Prompt.”** The main experiment should test whether task/subtask text, visual subgoals and execution metadata let a policy learn from varied behavior and select desired behavior at inference. Compatibility with SmolVLA alone is insufficient. This supersedes treating PushT's visual-subgoal A/B comparison as the main π0.7 experiment in the initial reconstruction plan.

**Recommended starting point:** a small task subset from `lerobot/libero_10_subtask`, supplemented with LIBERO-Goal for instruction-sensitive choices and new simulator rollouts containing successful, slow, failed and recovered behavior. The released demonstrations alone do not meet the metadata requirement. Final task selection depends on validating subtask meanings, segment endpoints and the simulator interface.

- **PushT:** keep the completed format audit for optional pipeline debugging. Its single fixed task cannot test the main language/subtask claims.
- **LIBERO:** practical first choice because the local LeRobot clone has a simulator wrapper, language tasks and a dataset with subtask IDs. Add and annotate mixed-quality rollouts on the same selected tasks.
- **CALVIN:** strong alternative for language-labeled temporal intervals, multiple views and simulator evaluation; requires an adapter and subset selection. Quality/mistake annotations still need work. [Official dataset documentation](https://github.com/mees/calvin/blob/main/dataset/README.md).
- **BridgeData V2 / DROID:** candidates for real-robot data and generator adaptation. Their task-language/image-action data does not by itself supply the validated segment-level mistake and 1–5 quality labels needed here, and offline data alone cannot provide our simulator rollout evaluation. [BridgeData V2](https://bridgedata-v2.github.io/), [DROID schema](https://github.com/droid-dataset/droid/blob/main/docs/the-droid-dataset.md).

## Required data for each Section V component

| Paper component | Minimum useful toy data | What we need to verify or add |
|---|---|---|
| A. Subtask instructions | Different task instructions plus semantically labeled time intervals inside trajectories | Validate ID-to-text mapping against videos or simulator predicates; arbitrary temporal splits are not semantic annotations |
| B. Subgoal images | Images synchronized with actions/state, with valid segment endpoints | Sample within episode bounds; retain camera identity; do not label a failed endpoint as successful completion |
| C. Speed | Known sampling rate and duration, with multiple speeds for the same task | Bin durations within task; retain success/timeout status so short failures are not called fast successes |
| C. Quality | Meaningful variation within each selected task, including successful and unsuccessful behavior | Declare and validate a quality rubric; a terminal success bit is not automatically a five-level quality score |
| C. Mistakes | Intervals identifying incorrect execution, failed attempts and recovery | Use simulator events and reviewed annotations; injected noise or decreasing reward alone is not a reliable mistake label |
| D. Control mode | Explicit action semantics and consistent units/controller configuration | A single end-effector mode is an acceptable simplification; a constant tag does not test mode switching |
| E. Prompt dropout | Every field has a value or an explicit unavailable mask | Distinguish missing annotations from deliberately dropped context; never fill unknown mistakes with false |

Keep label provenance (human, simulator event, or heuristic), confidence and missingness. Split by episode and initial-state group before generating image pairs; account for checkpoint pretraining when claiming held-out generalization. Collect quality levels across matched tasks/initial-state distributions so metadata does not merely identify easier tasks.

**Main comparisons:** hold the trajectory mixture and policy-training budget fixed while adding subtask, subgoal and metadata conditioning, then ablate components from the full prompt. Separately run the paper's nested quality-subset experiment with and without metadata, keeping other prompt components matched. Evaluate prompt sensitivity and behavior on fresh simulator episodes, rather than relying only on action prediction error.

## Which benchmark fits the test

| | PushT | LIBERO | MetaWorld | CALVIN |
|---|---|---|---|---|
| Does a subgoal image carry information? | Little: one fixed goal | **Yes:** LIBERO-Goal has 10 goals in one scene | Little: the scene gives away the task | Yes |
| Multi-step tasks | No | **Yes:** LIBERO-10 | No | Yes: 5-instruction chains |
| Language matters | No | Yes | Weakly | Yes |
| Ready-made SmolVLA baseline | Community only, untested | **Official** `lerobot/smolvla_libero` | No | No |
| World model difficulty | Easy: 96×96, 2-D | Harder: 3D scenes, 256×256 | Hard: 480×480 | Harder |
| In LeRobot | Yes | Yes | Yes | No |
| Role | Optional pipeline debugging | **Primary candidate, plus labeled rollouts** | Not selected | Semantic-segmentation alternative |

## PushT

All three versions cover the same 206 human demos: 25,650 frames at 10 fps, one task ("push the T onto the target"), a 2-D agent-position state and a 2-D action (target position).

| Dataset | Format | Download | Extra fields | Use |
|---|---|---|---|---|
| [lerobot/pusht](https://huggingface.co/datasets/lerobot/pusht) | 96×96 video | 0.01 GB | `next.reward`, `next.success` (semantics need validation) | Optional pipeline debugging |
| [lerobot/pusht_image](https://huggingface.co/datasets/lerobot/pusht_image) | 96×96 images | 0.03 GB | Same as above | Not needed |
| [lerobot/pusht_keypoints](https://huggingface.co/datasets/lerobot/pusht_keypoints) | No images | < 0.01 GB | `observation.environment_state`: 16 numbers, the T's 8 keypoints (x, y) | **T position and angle for v2 phase labels** |

- **Verified label limitation:** the [actual-file audit](artifacts/pusht_preflight/dataset_report.json) found all stored `next.success` values false and maximum recorded reward 0.9488797. This does not establish that all demos failed. Do not directly derive quality buckets from these fields.
- Episode duration can be derived. Raw coverage, quality and mistake labels need validation; drops in overlap can be necessary for a successful rotation.
- Proposed approach/push/rotate/align phases would require a validated detector using agent position and T geometry. They are not supplied semantic annotations.
- **To verify:** that `pusht` and `pusht_keypoints` frames line up one-to-one by `episode_index` and `frame_index`. The counts match exactly.
- **Simulator:** `gym-pusht`, installed with LeRobot's `[pusht]` extra. Episodes run at 10 fps for at most 300 steps.
- **Image size check:** LeRobot's PushT env config renders 384×384, while the dataset is 96×96. Confirm what reaches the policy.
- **Extra data:** rollouts of your DOT checkpoints (see [model.md](model.md)) add mixed-quality episodes for the metadata add-on.

## LIBERO

All versions: 10 fps, two 256×256 cameras (scene view `image` and wrist view `image2`), 8-D state, 7-D action.

| Dataset | Format | Download | Content | Use |
|---|---|---|---|---|
| [lerobot/libero](https://huggingface.co/datasets/lerobot/libero) | Video | 1.94 GB | All 4 suites: 40 tasks, 1,693 demos, 273,465 frames | **LIBERO-Goal, Spatial and Object** (filter by task) |
| [lerobot/libero_10](https://huggingface.co/datasets/lerobot/libero_10) | Video | 0.65 GB | LIBERO-10 (two-step tasks) | Superseded by the next row |
| [lerobot/libero_10_subtask](https://huggingface.co/datasets/lerobot/libero_10_subtask) | Video | 0.83 GB | LIBERO-10: 500 demos, 138,090 frames, plus a per-frame `subtask_index` | **Primary candidate**: validate ID meanings and boundaries, then add mixed-quality rollouts |
| [lerobot/libero_goal_image](https://huggingface.co/datasets/lerobot/libero_goal_image) | Images | 6.31 GB | LIBERO-Goal | Not needed |
| [lerobot/libero_spatial_image](https://huggingface.co/datasets/lerobot/libero_spatial_image) | Images | 6.60 GB | LIBERO-Spatial | Not needed |
| [lerobot/libero_object_image](https://huggingface.co/datasets/lerobot/libero_object_image) | Images | 9.25 GB | LIBERO-Object | Not needed |
| [lerobot/libero_10_image](https://huggingface.co/datasets/lerobot/libero_10_image) | Images | 12.77 GB | LIBERO-10 | Not needed |
| [lerobot/libero_10_image_subtask](https://huggingface.co/datasets/lerobot/libero_10_image_subtask) | Images | 16.02 GB | LIBERO-10 with subtask ids | Not needed |
| [HuggingFaceVLA/libero](https://huggingface.co/datasets/HuggingFaceVLA/libero) | Images | 34.93 GB | All 4 suites | Not needed |

There are no separate video datasets for LIBERO-Goal, Spatial or Object; filter `lerobot/libero` by task instead.

**`libero_10_subtask` vs what `smolvla_libero` was trained on** (checked 2026-10-28). `lerobot/smolvla_libero` was fine-tuned on `lerobot/libero` (25k steps, batch 32). That set's LIBERO-10 portion differs from our download:

| | `lerobot/libero`, LIBERO-10 part | `lerobot/libero_10_subtask` (downloaded) |
|---|---|---|
| Episodes | 379 (29–49 per task) | 500 (50 per task) |
| Frames | 101,469 | 138,090 |
| Median episode length | Slightly shorter (e.g. 181 vs 187 for the book task, 401 vs 409 for both moka pots) | – |
| Subtask ids | No | Yes |
| Cameras, state, action, fps | Two 256×256 cameras, 8-D state, 7-D action, 10 fps | Same |

- `lerobot/libero`'s total of 1,693 episodes matches the filtered LIBERO releases (failed replays and idle steps removed). The count match is the only evidence; it isn't confirmed.
- Whether its 379 episodes are a subset of our 500 also isn't confirmed.
- **Camera keys differ.** `smolvla_libero` expects `observation.images.camera1` / `camera2`; the datasets use `image` / `image2`. Map them, and check which camera is which before training.
- **Fair comparison.** `smolvla_libero` has already seen LIBERO-10 demos. Train baseline A and π0.7-style B on the same `libero_10_subtask` data for the same number of steps, so B's gain can't come from extra data.

**Local copy:** `data/libero_10_subtask` (834 MB, 223 files, 500 episodes, 10 tasks, 138,090 frames; checksums verified).

**Subtask labels: what the data contains** (all 500 episodes, plus a visual check of start and end frames, 2026-10-28)

- **16 subtask ids (0–15), shared across tasks.** The same id marks the same step in different tasks. The dataset stores ids only, not text.
- **The order is fixed per task.** Each task follows one id sequence in almost every episode (table below).
- **Real step counts:** 8 tasks have 2 steps. Task 9 (book) is a single segment with no split. Task 3's first segment (id 15) is always 1 frame long, a labelling artifact, so it really has 2 steps: id 5, then id 4.

| # | Task | Sequence in data (episodes) | Subtask text (id) |
|---|---|---|---|
| 0 | turn on the stove and put the moka pot on it | 15 → 4 (50) | turn on the stove (15) · put the moka pot on the stove (4) |
| 1 | put the black bowl in the bottom drawer of the cabinet and close it | 8 → 2 (50) | put the black bowl in the bottom drawer (8) · close the drawer (2) |
| 2 | put the yellow and white mug in the microwave and close it | 14 → 1 (50) | put the yellow and white mug in the microwave (14) · close the microwave (1) |
| 3 | put both moka pots on the stove | 15 → 5 → 4 (47); 15 → 5 (2); 15 → 4 (1) | [1-frame artifact (15)] · put the first moka pot on the stove (5) · put the moka pot on the stove (4) |
| 4 | put both the alphabet soup and the cream cheese box in the basket | 9 → 12 (50) | put the alphabet soup in the basket (9) · put the cream cheese box in the basket (12) |
| 5 | put both the alphabet soup and the tomato sauce in the basket | 9 → 13 (50) | put the alphabet soup in the basket (9) · put the tomato sauce in the basket (13) |
| 6 | put both the cream cheese box and the butter in the basket | 12 → 11 (50) | put the cream cheese box in the basket (12) · put the butter in the basket (11) |
| 7 | put the white mug on the left plate and put the yellow and white mug on the right plate | 6 → 7 (50) | put the white mug on the plate (6) · put the yellow and white mug on the right plate (7) |
| 8 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 6 → 3 (49); 3 → 6 (1) | put the white mug on the plate (6) · put the chocolate pudding to the right of the plate (3) |
| 9 | pick up the book and place it in the back compartment of the caddy | 10 (49); 0 (1) | whole task, one segment (10, or 0 in one episode) |

**Provenance.** Ids 0–8, 10, 14 and 15 were identified from the start and end frames: the stove turning red, doors and drawers closing, items arriving. Ids 9, 11, 12 and 13 (the basket items) come from which tasks share each id; the frames show items going into the basket, but their labels can't be read at thumbnail size. Segment boundaries were checked at start and end frames only, not frame by frame. Contact sheets, the script and the per-id evidence are in the [visual check](artifacts/section_v_dataset_audit/libero_10_subtask_visual/README.md).

**Independent sample audit:** inspected episodes 0, 50, …, 450 (one for each task) at revision `66a4f67dd88bde16c2d55395edf23617d04972dd`. The stove-and-pot sample changes IDs 15→4; bowl-and-drawer changes 8→2; the book sample uses only ID 10 throughout. The two-pot sample begins with a one-frame ID 15 segment, then 5→4. Verify unusual short segments before treating them as world-model targets. This was a structural sample audit, not a visual verification. [Audit details](artifacts/section_v_dataset_audit/libero_10_subtask/annotation_audit.json).

**Metadata gap:** the inspected schema has no quality, mistake or success fields. The official LIBERO release describes high-quality demonstrations; it does not provide the controlled quality mixture this experiment needs. Add rollouts from several policy checkpoints on the same tasks, retain failures and recoveries, and annotate them using the declared rubric. [Official LIBERO dataset description](https://github.com/Lifelong-Robot-Learning/LIBERO#datasets).

- **Simulator:** LeRobot's `envs/libero.py`, installed with the `[libero]` extra. Heavier than PushT.
- **Baselines:** `lerobot/smolvla_libero` (official) and `lerobot/pi05_libero_finetuned` (see [model.md](model.md)).

## MetaWorld (not used)

| Dataset | Format | Download | Content |
|---|---|---|---|
| [lerobot/metaworld_mt50](https://huggingface.co/datasets/lerobot/metaworld_mt50) | 480×480 images | 45.86 GB | 50 tasks, 2,500 demos, 204,806 frames at 80 fps; 4-D state and action; 39-D environment state |

MetaWorld was the first plan because its scripted experts give unlimited labelled data. It was dropped: each task's scene gives the task away, the tasks are single-step, and this dataset is large. The scripted experts remain available through LeRobot's `envs/metaworld.py` if needed.

## CALVIN (alternative)

CALVIN is relevant to Section V: `auto_lang_ann.npy` supplies language/task IDs with start/end indices; frames contain static and gripper RGB, proprioception, scene state, and absolute/relative end-effector actions. Quality and mistake labels still require a documented annotation procedure; play data is not automatically labeled mixed-quality data. An episode-level overall task may also need defining when using short language-labeled play segments.

The official archives are large (D→D: 166 GB; ABC→D: 517 GB). The 1.3 GB debug set is useful to verify an adapter, not to establish benchmark results. A verified subset conversion is another option. [Official format and downloads](https://github.com/mees/calvin/blob/main/dataset/README.md).

## Sources

- Hugging Face dataset pages linked in the tables. Sizes, episode counts, frame counts and fields were read via the Hugging Face API (`meta/info.json`) on 2026-10-28.
