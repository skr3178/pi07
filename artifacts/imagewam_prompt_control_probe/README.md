# ImageWAM prompt controls and chaining diagnostics

Date: 2026-10-29. Follow-up to the [original subgoal probe](../imagewam_subgoal_probe/README.md).

**Result:** the released checkpoint remains a usable candidate for short-horizon visual prediction, but this probe does not demonstrate reliable instruction-controlled goals. Opposing instructions usually produce nearly the same continuation. Holding the initial robot state fixed preserves the drawer/pudding chains surprisingly well; the microwave chain still fails. These conclusions are limited to the six inspected states.

## Question and design

Does the released LIBERO checkpoint change its predicted outcome when the instruction changes, with image, robot state, seed and sampling settings held fixed? Separately, how much of the previous chaining result survives when future demonstration states are unavailable?

- Six inputs: the original three frames, plus a frame 32 steps before each annotated subtask endpoint. Two cameras and the current 8-D state are supplied for every single-step call.
- Five text conditions: full task, aligned subtask, aligned paraphrase, conflicting subtask, and empty user text. Empty text still passes through Qwen's normal chat template; it is not a zero embedding or a trained unconditional branch.
- Three matched seeds per condition: 90 single-step generations. An identical-input/seed repeat checks reproducibility.
- Three additional chains use the original inputs and aligned subtask prompts, holding the initial state fixed for all 6–7 calls. No future state or image is supplied to these chains. The chain lengths are chosen using the known demonstration segment endpoints; this is not an autonomous stopping rule. Holding state fixed deliberately creates a possible state/image mismatch and is only a stress test.
- Existing weights and environment only; network model loading disabled. Settings match the earlier probe: 224 × 448, BF16, 10 denoising steps, shift 5, no CFG.

Exact instructions, states, frame indices and reference paths are in [manifest.json](manifest.json). Inputs were visually inspected before inference. The conflicting instructions are open versus close for the drawer/microwave, and left versus right placement for the pudding. These instructions may be outside the checkpoint's training distribution; a negative result does not prove the architecture cannot use language.

## Measurements

Report full-image, moving-region and static-region PSNR/MAE separately for each camera. A moving pixel changes by more than 20/255 in at least one channel between the current frame and the demonstration's t+16 frame. This mask measures image change, including wrist-camera movement; it is not an object segmentation.

For task/aligned/paraphrase/empty prompts, compare to the demonstration and to copying the input. Counterfactual outputs have no matching ground truth and are **not** scored as accurate or inaccurate against the original continuation. Their differences from aligned outputs measure sensitivity only. Compare those differences with same-meaning paraphrases and changes caused by random seeds. Inspect all seeds for whether the requested object relation actually changes.

Compare frozen-state chains with the previous oracle-state chains and both the real chain endpoint and semantic segment endpoint. Check whether the first prediction reproduces the earlier probe before attributing chain differences to the state input.

## Results: instruction sensitivity

All 90 single-step predictions completed. The six per-case sheets include every seed and both cameras; the overview sheets use seed 0 without selection. Visual inspection found no consistent reversal of the requested object outcome across matched seeds. The drawer generally continues closing under an opening instruction, microwave predictions remain similar for opening and closing, and pudding left/right predictions are nearly identical. This is a qualitative assessment, not a simulator-validated success rate.

Main-camera sensitivity, measured as mean absolute pixel difference on the 0–255 scale:

| Input | Opposing prompt, same seed | Paraphrase, same seed | Same aligned prompt, different seeds |
|---|---:|---:|---:|
| Drawer, original t=160 | 1.01 | 0.73 | 3.06 |
| Drawer, late t=228 | 1.51 | 0.77 | 11.17 |
| Microwave, original t=212 | 1.81 | 1.44 | 6.47 |
| Microwave, late t=296 | 3.48 | 2.41 | 9.35 |
| Pudding, original t=104 | 0.35 | 1.46 | 11.51 |
| Pudding, late t=170 | 0.29 | 0.77 | 8.53 |

Opposing-prompt differences are smaller than between-seed differences for all six inputs in **both** cameras. The same is true within the demonstration-defined moving regions (see the full report). For pudding, changing left/right has less effect than a same-meaning paraphrase. Neither this pixel metric nor seed variability alone proves semantic failure; the image inspection is needed to assess the object relation.

Empty text changes outputs more than the opposing instruction does for every input/view. In the late pudding case, aligned-text main-camera PSNR is 20.90 dB versus 17.95 dB with empty text and 17.46 dB for copying. The model therefore is not completely insensitive to language. The narrower conclusion is that the tested **instruction distinctions** do not reliably override the visual continuation. Correctly encoded opposing prompts were confirmed to have distinct, finite Qwen embeddings; the empty control contains the normal chat-template tokens.

These are also not counterfactual ground-truth comparisons: the original demonstration is only a reference for its own instruction. Some opposing commands may be outside training, and the two commanded trajectories could share an initial approach within a short horizon. We cannot infer universal language blindness from this sample.

## Results: chains without future state

All three original aligned seed-0 first steps reproduce the earlier probe **pixel-for-pixel**, making the changed state input a controlled difference for the subsequent chain. One same-input/seed repeat within this run is also pixel-identical.

Main-camera PSNR versus the annotated segment endpoint:

| Case | Copy initial image | Fixed initial state | Earlier oracle future state |
|---|---:|---:|---:|
| Drawer | 13.78 | 15.93 | 15.89 |
| Microwave | 14.57 | 14.86 | 14.57 |
| Pudding | 14.67 | 18.15 | 18.26 |

Drawer and pudding retain similar visual progress with the initial state held fixed. The supplied future states were **not essential to these two chain examples**, correcting the suspicion that their appearance was mainly explained by oracle state. This does not establish state irrelevance in general or make a fixed-state rollout physically consistent.

Microwave fails with either state scheme: the door/robot geometry becomes implausible and the wrist view loses the scene structure. Its fixed-state wrist PSNR is 7.92 dB, below the 8.76 dB copy baseline; the earlier oracle chain scored 11.37 dB. Small main-camera score gains should not be read as successful closure.

Only one seed was used for each chain, and there is no closed-loop robot test. The predictions do not establish reachability by SmolVLA.

## Runtime and verification

- **110 generations:** 90 prompt controls, 1 repeated prediction and 19 chained predictions.
- **1.386 s/image mean**, 1.407 s/image at the 95th percentile, including image encoding/decoding and transfer of the generated image back to CPU.
- **9.51 GB peak PyTorch allocation** after releasing Qwen3; 9.95 GB peak reserved memory. The allocated peak while loading/using the text encoder was 17.61 GB. These are decimal GB, not total device usage.
- Model loading took 7.8 s on this run; the earlier cold-drive run took 108 s. Do not assume the faster loading figure for a fresh machine.
- Local RTX PRO 4000 Blackwell, torch 2.7.1+cu128. No new model downloads, environment changes or training.
- MoT checkpoint loaded with zero missing/unexpected keys, and the proprio encoder loaded. Repeated prediction and all three prior-probe comparisons were pixel-identical.

See [runtime.json](results/runtime.json), [conditioning audit](results/conditioning_audit.json), [full metrics](results/report.json) and [run.log](run.log).

## Decision for the π0.7 toy

Keep ImageWAM as a pretrained visual-prediction baseline and possible initialization. Its scene preservation and runtime are suitable for further work, but these tests do not justify treating it as a verified language-steerable, end-of-subtask generator.

For the Section V objective, the next adaptation should pair **current images + explicit subtask text** with **validated segment-end images**, and include situations where language selects different outcomes from similar visual states. Endpoint fine-tuning on a single fixed continuation per scene may simply preserve the current bias. LIBERO-Goal or suitable alternate simulator rollouts would make this distinction more testable. LoRA memory requirements and the benefit to a goal-conditioned SmolVLA policy remain unmeasured.

## Inspect the outputs

- [Prompt controls: main camera](results/prompt_controls_main.png) and [wrist camera](results/prompt_controls_wrist.png).
- [Chaining: main camera](results/chains_main.png) and [wrist camera](results/chains_wrist.png).
- All-seed sheets: [drawer original](results/drawer_close_original_all_seeds.png), [drawer late](results/drawer_close_late_all_seeds.png), [microwave original](results/microwave_close_original_all_seeds.png), [microwave late](results/microwave_close_late_all_seeds.png), [pudding original](results/pudding_place_original_all_seeds.png), [pudding late](results/pudding_place_late_all_seeds.png).
- [Input selection](results/inputs_main.png), [manifest](manifest.json), [script](run_probe.py), and individual images under `results/predictions/`.

## Reproduction

Run from `/home/skr/Downloads/pi_07`, using the installed environment:

```bash
PYTHONDONTWRITEBYTECODE=1 /media/skr/storage/envs/imagewam-venv/bin/python pi07_code/artifacts/imagewam_prompt_control_probe/run_probe.py prepare
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/media/skr/storage/hf_home /media/skr/storage/envs/imagewam-venv/bin/python -u pi07_code/artifacts/imagewam_prompt_control_probe/run_probe.py run > pi07_code/artifacts/imagewam_prompt_control_probe/run.log 2>&1
PYTHONDONTWRITEBYTECODE=1 /media/skr/storage/envs/imagewam-venv/bin/python pi07_code/artifacts/imagewam_prompt_control_probe/run_probe.py analyze
```

The script writes only within this artifact directory. The CPU prompt cache allows later reuse without loading Qwen3. This is an inspection of six states from three demonstration episodes, not a held-out benchmark or a closed-loop policy evaluation. Training-data overlap is unresolved.
