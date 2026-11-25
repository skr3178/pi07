# Running the local π0.7 mechanism study

The current state, measurements and pending gates are in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). This implements a small
SmolVLA/ImageWAM study of the paper's prompt mechanisms; it does not reproduce
π0.7's model scale, robot diversity or reported performance.

## Environment and assets

Run commands from `/home/skr/Downloads/pi_07`. Policy training/evaluation uses
`/media/skr/storage/envs/pi07-lerobot/bin/python`; the ImageWAM worker starts
its own `/media/skr/storage/envs/imagewam-venv/bin/python` process. Existing
model/data paths are recorded in the plan and individual manifests. No model
download is required. GPU jobs need host GPU access; the application sandbox
does not expose CUDA. Existing Hugging Face dataset caches also create lock
files outside the workspace when reading data.

[Environment records](artifacts/reproduction_environment_20261104/) include
package versions, editable-install provenance and repository commits/status.
Each training/evaluation run saves its input schedule, checkpoint/source
hashes, report and source snapshot. Snapshots preserve source for review;
restore them to the project root to reproduce a historical revision, since
the code resolves project assets relative to that root.

## Training and scoring

The five profiles are `task`, `subtask`, `goal`, `subtask_goal`, and `full`.
All use two current cameras, eight state values, seven action dimensions,
50 predicted actions and eight executed actions. They share a checkpoint,
normalizers, schedule and optimization budget. `full` includes available
duration/quality/mistake labels. Unknown annotations remain absent.

For the controlled LIBERO-10 text/image factorial:

```bash
/media/skr/storage/envs/pi07-lerobot/bin/python pi07_code/train_section_v.py \
  --profile subtask_goal --subtask-drop-given-goal 0 \
  --output pi07_code/artifacts/NEW_RUN --eval-every 250
```

Default goals appear in 25% of training samples. `full` defaults to 30%
conditional subtask dropout, 15% all-metadata dropout and 5% individual-field
dropout. To isolate metadata, train both `subtask_goal` and `full` with the
same 30% conditional text dropout on the curated mixed dataset:

```bash
/media/skr/storage/envs/pi07-lerobot/bin/python pi07_code/train_section_v.py \
  --dataset mixed --dataset-root pi07_code/data/libero10_metadata_mixture_20261104 \
  --annotations pi07_code/data/libero10_metadata_mixture_20261104/annotations_v2.jsonl \
  --schedule-dir pi07_code/artifacts/metadata_mixture_schedule_20261104 \
  --profile full --seed 20261104 --eval-every 250 \
  --output pi07_code/artifacts/NEW_METADATA_RUN
```

For LIBERO-Goal use `--dataset libero_goal` and
`--schedule-dir pi07_code/artifacts/libero_goal_schedule_20261104`. Its
FastWAM gripper targets are converted with `1 - 2 * raw`; state and arm
commands are unchanged. It has no intermediate subtask annotations.

Saved live experiments are in `experiments/section_v_*_development.json`.
The evaluator rejects real demo futures online, mismatched checkpoint layouts,
unseen quality/mistake requests and reserved final starts. It records full
reset states, actions, predicates, images, generation inputs and timing.

```bash
/media/skr/storage/envs/pi07-lerobot/bin/python pi07_code/evaluate_libero.py \
  --spec pi07_code/experiments/section_v_l10_development.json \
  --output pi07_code/artifacts/NEW_EVALUATION
/media/skr/storage/envs/pi07-lerobot/bin/python pi07_code/analyze_libero_evaluation.py \
  --run pi07_code/artifacts/NEW_EVALUATION
```

Real-future/copy/wrong-task/offline action-error controls use
`evaluate_section_v_offline.py --run TRAINING_RUN --output NEW_OUTPUT`.
For annotated runs also supply the original `--annotations` file. Offline
action error is distinct from simulator success.

## Resume and resource use

Launch long jobs through `background_job.py start --job-dir NEW_JOB_DIR -- COMMAND`.
Run the launcher with host process access so detached workers persist.
`background_job.py status --job-dir JOB_DIR` checks PID identity as well as
the saved status. Do not start a duplicate based only on an old log.

Training saves immutable intermediate checkpoints (every 500 updates by
default), optimizer and RNG state. `--resume CHECKPOINT` uses a new output
directory and must preserve the model, optimizer, source and schedule prefix.
An exact two-update interrupted/uninterrupted check is saved. Simulator
`--resume` reruns only missing/failed episodes while preserving earlier source
and reset records.

ImageWAM startup peaks around 17.6 GB before unloading Qwen; each policy
training process currently occupies roughly 6 GB. Do not start ImageWAM while
two training jobs are resident. Runtime is synchronous, so measured generator
latency is not a real-time-control claim.

## Interpretation limits

- A generated-goal improvement must beat the copied-current control, not only
  the same model with goal inputs absent.
- Quality labels in the curated mixture are provisional assistant visual
  judgments. Sparse mistake labels cover 64 frames; all other frames remain
  unknown. There is no independent annotation agreement estimate.
- LIBERO-10 commonly has one demonstrated order per scene. ImageWAM usually
  continues that order; reliable counterfactual subtask steering is unproven.
- LIBERO-Goal tests familiar instructions. Its task endpoints are not annotated
  intermediate boundaries. Filtered-frame time is not certified physical time.
- Development uses starts 0/1; collection uses 12–49. Reserved starts 2–11
  remain unused until choices are frozen. Pretraining overlap and demo/rollout
  physical-start independence are not certified.
- Negative results are retained. Functional integration and short smoke checks
  do not establish the paper's scientific claims.
