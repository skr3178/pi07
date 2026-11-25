# ImageWAM FLUX.2 4B LIBERO: subgoal-image probe (2026-10-29)

Can ImageWAM's image path stand in for π0.7's world model, which produces subgoal images? It ran with the released LIBERO-trained checkpoint, without additional fine-tuning. The setup is in [`../imagewam_setup/README.md`](../imagewam_setup/README.md). The [follow-up controls](../imagewam_prompt_control_probe/README.md) test conflicting instructions, empty text and chaining without future demonstration states.

## What the image path does

`infer_video_flux2` takes the current frame (main camera | wrist camera, 224×448), a prompt and the 8-dim robot state. It predicts a possible observation **16 dataset steps ahead**. This is 1.6 seconds in our dataset's 10-fps timeline; the checkpoint's training data had no-op steps removed, so it is not necessarily a fixed elapsed physical time. Its image target is the final frame of a 17-frame window (`endpoint_frames_only: true`, `num_frames: 17`, stride 1), using LIBERO task text. The complete model also trains an action head. This probe uses 10 denoising steps without classifier-free guidance. π0.7's world model instead targets the **end of the current subtask** and is conditioned on subtask text.

## Tests

| | Input | Scored against |
|---|---|---|
| A. Single step | 3 LIBERO cases (the same frames as the BAGEL probe) × {task, subtask} prompt × 3 seeds | Real frame at t+16; "copy the current frame" baseline |
| B. Chained | Each prediction fed back in 6–7 times to reach near the subtask end. **Uses the real dataset state at every step (oracle).** | Real frame at t+16k and the real subtask-end frame |
| C. π0.7 food steps | 5 real-robot images from the π0.7 website; one camera, wrist slot mid-gray, state set to zero (mid-range) | π0.7's own subgoal images (qualitative) |

Caveat: ImageWAM was trained on all of LIBERO-10, so the LIBERO episodes here are probably in its training set.

## Results

**Cost:** 1.38 s per image; 9.5 GB peak PyTorch GPU allocation without Qwen3 (17.5 GB while Qwen3 encoded the prompts). These are decimal GB of allocated tensors, not total device usage. Loading takes 108 s from the Seagate drive. The MoT checkpoint loaded with 0 missing and 0 unexpected keys; the proprio encoder also loaded successfully.

**A. Single step (PSNR in dB, mean of 3 seeds; "moving" = pixels that change by >20/255 between t and t+16, pooled across both cameras; "main" is the full main camera):**

| Case | Copy baseline: main / moving | Task prompt: main / moving | Subtask prompt: main / moving |
|---|---|---|---|
| drawer_close | 20.9 / 9.8 | 21.1 / 10.7 | 21.0 / 10.6 |
| microwave_close | 16.7 / 7.8 | 16.4 / 9.6 | 16.6 / 9.6 |
| pudding_place | 15.8 / 9.4 | 19.6 / 13.0 | 18.3 / 13.3 |

- The single-step frames preserve scene and robot identity much better than the base BAGEL examples. Visual plausibility alone does not establish physical correctness or useful policy goals.
- The predicted change is generally small. It clearly beats copying on pudding_place and marginally on drawer_close. Full-image PSNR is worse than copying on microwave_close in both views, although pooled moving-region PSNR improves in all three cases.
- Task and subtask prompts produce broadly similar outputs here. Because the prompts request compatible behavior, this does not establish that the model ignores text.

**B. Chained to near the subtask end (main camera PSNR vs the real subtask-end frame; seed 0):**

| Case | Steps | Copy baseline | Chain, task prompt | Chain, subtask prompt |
|---|---|---|---|---|
| drawer_close | 6 | 13.8 | 16.0 | 15.9 |
| microwave_close | 7 | 14.6 | 15.0 | 14.6 |
| pudding_place | 6 | 14.7 | 17.9 | 18.3 |

- drawer_close and pudding_place: the chained images roughly resemble the subtask endpoints, including a closed drawer and a pudding box visible beside the plate in the wrist view. Arm pose and detailed geometry are not exact matches.
- microwave_close: fails. The door is drawn as a floating open panel, and the wrist view degrades into flat surfaces.
- These are diagnostics with oracle future state: the model is told the demonstration's arm state at every step. A deployed system would not have those states. They are not a formal upper bound, because generated images can also become inconsistent with the supplied states.

**C. π0.7 food steps:** the outputs largely copy the inputs, with appearance changes and no clear completion of the instructed actions. These are real-robot scenes outside LIBERO training and also lack valid wrist observations and proprioception; this test cannot isolate visual-domain transfer. The website's reference subgoals are generated examples, not ground-truth robot futures.

## What it means for the toy π0.7

**Follow-up completed 2026-10-29:** opposing prompts produced weak differences relative to random-seed variation in six inputs, while empty text had a larger effect. Fixed-initial-state chains retained the drawer/pudding results, so future state was not essential to those two examples; microwave still failed. See the [controlled follow-up and all-seed grids](../imagewam_prompt_control_probe/README.md).

- The released checkpoint supplies plausible **16-step future-image candidates** on these LIBERO examples. Their utility as policy subgoals and their response to conflicting instructions still require testing. This probe does not demonstrate π0.7-style end-of-subtask generation in one call.
- Ways forward:
  1. **Fine-tune the image path** on `libero_10_subtask`, with the target set to the segment's end frame and the subtask text as the prompt. This mirrors what π0.7 did to BAGEL. A full fine-tune of 4B doesn't fit in 24 GB; the repo's built-in LoRA option might.
  2. **Evaluate the 16-step candidates as subgoals**, replanning from actual observations. This departs from the paper but needs no world-model training. At 1.38 s per generation, synchronous refresh every 8 steps at 10 Hz would miss that 0.8 s interval; account for latency explicitly.
  3. **Diagnose chaining without oracle state.** Holding the initial state fixed is a stress test, not a realistic state predictor. Advancing state from predicted actions would require a simulator or a validated controller/dynamics model, rather than simply summing action values. See the follow-up linked above.

## Files

- `run_probe.py`: the whole probe. Run with the ImageWAM venv, `HF_HOME=/media/skr/storage/hf_home`, and `ImageWAM/.env.local` sourced.
- `preview_first_case.py`: a 5-column preview for one case (`results/drawer_close_preview.png`).
- `results/libero_main_camera.png`, `results/libero_wrist_camera.png`: all LIBERO cases. Columns: current | real t+16 | pred (task) | pred (subtask) | chain (task) | chain (subtask) | real chain end | subtask end.
- `results/pi07_food.png`: current | ImageWAM | π0.7 subgoal.
- `results/report.json`: every per-seed number. `run.log`: full log.
