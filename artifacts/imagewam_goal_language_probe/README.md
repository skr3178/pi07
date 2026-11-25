# ImageWAM language test on LIBERO-Goal start frames (2026-10-29)

Follow-up to the [subgoal probe](../imagewam_subgoal_probe/README.md) and the [prompt-control probe](../imagewam_prompt_control_probe/README.md). Those tested mid-motion LIBERO-10 frames, where opposite instructions barely changed the prediction. This test asks: **does ImageWAM follow the instruction when the image alone can't decide the outcome?**

## Setup

- **Data:** LIBERO-Goal from ImageWAM's own training set (`yuanty/LIBERO-fastwam`, folder `lerobot_v30/libero_goal_no_noops_lerobot`, 0.8 GB, at `/media/skr/storage/datasets/LIBERO-fastwam`). These are the same 512 px renders the model trained on, so there is no rendering mismatch.
- **Start frames:** frame 0 of 3 episodes (35: open middle drawer, 145: bowl on plate, 407: turn on stove). The arm is at home and nothing has moved yet. All 10 LIBERO-Goal tasks share this kitchen.
- **Prompts:** the 10 LIBERO-Goal instructions, exactly as in training, plus empty text.
- **Generation:**
  - Single step (16 steps ahead), seeds 0–2.
  - Chain of 5 calls (80 steps ahead), seeds 0–1. The robot state is held at frame 0, so no future information is used.
  - 363 images at 1.40 s each.
- **Scoring:** reference-image task classification. Each prediction is compared, at coarse resolution with both cameras, to real frames at the same nominal horizon from 5 other episodes of each task. The task with the lowest mean squared pixel error, averaged over its 5 references, is its "predicted task".
  - Chance: 10%.
  - Real-frame reference score: the same check on real frames, leaving each queried episode out of its class's references. These frames are held out of their own classification reference set, not out of ImageWAM's training. This is not a performance ceiling: generated images can be more stereotyped and easier to classify than real images.

## Results

| Nominal horizon | Reference-image task match | Real-frame reference score | Distinct tasks predicted per 10 instructions | Difference between instructions (same seed) | Difference between seeds (same instruction) |
|---|---|---|---|---|---|
| 16 steps (1 call) | **0.70** | 0.74 | 7.4 | 5.35 | 3.88 |
| 80 steps (5 calls) | **0.98** | 0.94 | 9.8 | 12.0 | 7.7 |

(Differences are mean absolute pixel differences on the main camera, 0–255 scale.)

- **It follows language here.** After 5 calls, 59 of 60 predictions match the instructed task. The images confirm it (`results/start_ep145_main.png`):
  - "open the top drawer…" opens the top drawer;
  - "turn on the stove" makes the burner glow red;
  - the wine bottle goes to the rack or onto the cabinet, as instructed;
  - the bowl goes to the plate, stove or cabinet top, as instructed;
  - each one closely resembles the real frame from another episode of that task.
- At 16 steps, 13 of the 27 misses are confusion among the three bowl-placement tasks (plate / stove / cabinet top). Other misses include confusion with the cream-cheese-in-bowl task. Similar early reaches make short-horizon classification difficult; the real-frame reference score is 0.74.
- Unlike the mid-motion LIBERO-10 test, changing the instruction now changes the image **more** than changing the seed.
- Empty text produces a generic reach. It isn't any one task consistently.

## How this fits with the earlier tests

ImageWAM follows the familiar LIBERO-Goal instructions in these starting scenes. In the earlier LIBERO-10 probe, opposing prompts barely changed the next prediction. Motion already underway is a plausible explanation for that difference, but this comparison does not isolate it: the task suite, prompt wording, rendering source and prediction horizon also differ. The earlier opposite prompts were not the checkpoint's exact training task strings. A matched start-versus-mid-motion test is needed to attribute the difference specifically to motion state.

For π0.7-style use this is encouraging, but Section VII refreshes goals when the subtask instruction changes **or after 4 seconds, whichever happens first**, using asynchronous generation. Refreshes can therefore happen mid-motion; a subtask boundary also does not guarantee that the arm is at rest. The paper's world model predicts segment-end images, so its 4-second refresh interval is not a required prediction horizon.

The training dataset metadata specifies 20 fps, and the checkpoint uses 17 frames at stride 1: one prediction spans 16 retained dataset transitions, nominally 0.8 seconds. Five chained calls have a nominal horizon of 80 transitions / 4 seconds. These are no-op-filtered trajectories, and generated chains have no measured physical timestamps, so these conversions do not establish exact elapsed simulator time. Our LIBERO-10 subtask dataset separately labels its videos as 10 fps; cross-dataset step alignment needs separate evidence. Five calls cost about 7 seconds of compute, so this chain cannot supply a fresh goal every 4 seconds of wall time on the tested setup. Offline evaluation or a slower refresh schedule remains possible.

## Limits

- Training episodes and the exact training instructions. New phrasings (for example our subtask strings) and unseen starting layouts are untested.
- 3 start scenes, at most 3 seeds, and a coarse pixel-based check. The images agree with it, but it's not a success rate from a simulator.
- Chaining holds the initial robot state fixed while the generated arm moves. This avoids future-state leakage but creates an image/state mismatch; it is a useful image-generation probe, not a validated dynamics rollout.
- Start-of-episode only. A test at LIBERO-10 subtask boundaries with subtask text is the case closest to π0.7's use and hasn't been run.

## Review verification (2026-10-29)

Recomputed every saved prediction's task classification from the PNGs and decoded dataset references, without rerunning the GPU model. All stored predicted labels reproduced: **63/90** generated matches and **37/50** real-frame matches at 16 transitions; **59/60** and **47/50** respectively at the nominal 80-transition horizon. The inspected main-camera grid supports instruction-dependent changes in drawer, burner and object placement. The original JSON keys containing `ceiling` are retained for provenance; interpret those values as reference scores.

**Assessment:** ImageWAM is the leading local candidate for goal images. This result supports familiar-task language control, while LIBERO-10 subtask wording, unseen layouts, recovery during motion and closed-loop policy benefit remain untested. Test the intended boundary prompts before deciding that further fine-tuning is needed.

## Files

- `run_probe.py`: everything (data selection, generation, scoring, grids).
- `results/start_ep{35,145,407}_{main,wrist}.png`: rows = instruction. Columns: 3 seeds at 16 steps | 2 chains at 80 steps | a real frame at 80 steps from another episode of that task.
- `results/report.json`: per-seed predicted tasks, accuracies, differences. `results/predictions/`: every image. `run.log`.

Run from this folder with `ImageWAM/.env.local` sourced:
`HF_HOME=/media/skr/storage/hf_home HF_HUB_OFFLINE=1 /media/skr/storage/envs/imagewam-venv/bin/python -u run_probe.py > run.log 2>&1`
