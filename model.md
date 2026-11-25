# Model options and checkpoints

**2026-11-04 local inventory and decisions:** [active implementation plan](IMPLEMENTATION_PLAN.md).
The current policy comparison starts from the local LIBERO-90 checkpoint and
uses its saved A/B-near/B-end adaptations; earlier entries below remain historical.

Catalog checked 2026-10-28; ImageWAM assessment updated 2026-10-29. Companion to [reconstruction.md](reconstruction.md) and [dataset.md](dataset.md).

The [LIBERO world model alternatives report](libero_world_model_alternatives.md) records the mimic-video, A2World, WoVR and RynnVLA-002 shortlist, their interfaces and pending checks.

**Local implementation update, 2026-10-29:** matched SmolVLA A/B checkpoints
have now been trained for 200 and 1,000 updates with frozen vision/language
weights and a trainable action expert/state projection. The explicit goal
adapter passes masking, pairing and padded-loss checks. At 1,000 updates,
offline action MSE is A 0.2892, B with real future goals 0.3403 and B with goals
dropped 0.2882; these numbers do not establish a benefit from goals. ImageWAM
is connected through a persistent worker using current simulator observations
and state only. See [training results](artifacts/goal_training_1000/README.md),
[live development results](artifacts/goal_closed_loop_dev/README.md) and the
[current plan](reconstruction.md). This remains a visual-guidance test; metadata
and subtask-text policy ablations have not run.

**Recommended picks:**

- **Policy, PushT:** fine-tune `lerobot/smolvla_base`, or start from `naonaon/smolvla_pusht` if it evaluates well.
- **Policy, LIBERO (main experiment):** start both arms from `lerobot/smolvla_libero` (Hugging Face fine-tuned it on `lerobot/libero`). Fine-tune it on `lerobot/libero_10_subtask` for the same number of steps twice: both A and B get identical full task text, while B additionally gets subgoal images. Test subtask text separately so the initial A/B isolates image guidance. Running `smolvla_libero` as-is gives an extra reference point, not the baseline.
- **World model:** Keep ImageWAM as a candidate predictor of usual LIBERO-10 continuations, not a validated steerable subtask generator. The boundary test did not establish reliable redirection. Test familiar-instruction control separately on LIBERO-Goal; see the [revised split plan](reconstruction.md). InstructPix2Pix (via GENIMA) or FLUX.2 Klein with LoRA remain adaptation options; a small diffusion UNet is the optional PushT route.
- **On disk:** ImageWAM FLUX.2 4B LIBERO and its image/text dependencies. Tested without additional fine-tuning (2026-10-29): scene-preserving 16-step predictions, but weak control by opposing subtask instructions in six inspected states. Empty text does affect the outputs, so this is not complete language insensitivity. See the [original probe](artifacts/imagewam_subgoal_probe/README.md) and [110-generation follow-up](artifacts/imagewam_prompt_control_probe/README.md). From three LIBERO-Goal start scenes it follows familiar instructions: 59/60 reference-image task matches after 5 chained calls ([language test and review](artifacts/imagewam_goal_language_probe/README.md)). This is not simulator success or held-out model generalization, and changes in task suite, wording and rendering prevent attributing the earlier weak control solely to mid-motion inputs. At LIBERO-10 subtask boundaries with our subtask wording it ignores the instruction and performs the scene's habitual step ([boundary test](artifacts/imagewam_boundary_probe/README.md)). BAGEL DFloat11 was tested and deleted on 2026-10-28.

"Repo size" below is everything in the Hugging Face repo. Some repos hold several weight formats or variants, so downloading one format takes less.

## Policy checkpoints: SmolVLA family

| Checkpoint | Trained on | Repo size | Downloads | Use here | On disk |
|---|---|---|---|---|---|
| [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base) | Community SO-100/SO-101 data (3 cameras, 6-D actions) | 0.91 GB | 177k | Starting point for any fine-tune | No |
| [naonaon/smolvla_pusht](https://huggingface.co/naonaon/smolvla_pusht) | `lerobot/pusht`, 100k steps from `smolvla_base`, batch 24 | 0.91 GB | 22 | PushT warm start and candidate baseline A. Documents the 1-camera, 2-D action overrides. No reported results. | No |
| [n3puiol/smolvla_pusht](https://huggingface.co/n3puiol/smolvla_pusht) | `lerobot/pusht`, 20k steps, not from `smolvla_base` | 0.73 GB | 68 | Older config format; may not load in current LeRobot | No |
| [crislmfroes/smolvla-pusht](https://huggingface.co/crislmfroes/smolvla-pusht) | `lerobot/pusht`, 5k steps | 1.20 GB | 14 | Barely trained | No |
| [lerobot/smolvla_libero](https://huggingface.co/lerobot/smolvla_libero) | `lerobot/libero` (official; 25k steps from `smolvla_base`) | 0.91 GB | 25k | **Starting point for both A and B** in the LIBERO test | Yes: `/media/skr/storage/models/smolvla_libero` |
| [HuggingFaceVLA/smolvla_libero](https://huggingface.co/HuggingFaceVLA/smolvla_libero) | LIBERO (official, earlier upload) | 1.22 GB | 39k | Alternative to the above | No |
| [lerobot/smolvla_libero_plus](https://huggingface.co/lerobot/smolvla_libero_plus) | LIBERO-Plus | 0.93 GB | 872 | Not needed | No |

All the PushT checkpoints use 50-step chunks executed whole, which is 5 s without re-planning at PushT's 10 fps. Evaluate them executing 8 of 50 steps.

## Baselines and comparison policies

| Checkpoint | What it is | Repo size | Use here | On disk |
|---|---|---|---|---|
| [lerobot/diffusion_pusht](https://huggingface.co/lerobot/diffusion_pusht) | Diffusion Policy on PushT (official) | 1.05 GB | Strong non-language reference point for PushT | No |
| [lerobot/vqbet_pusht](https://huggingface.co/lerobot/vqbet_pusht) | VQ-BeT on PushT (official) | 0.16 GB | Second reference point | No |
| DOT PushT policy ([skr3178/dot_resnet](https://github.com/skr3178/dot_resnet)) | Your Decoder-Only Transformer: ResNet-10, 8 layers, LoRA rank 16, chunk 20 | 147 MB per checkpoint | Mixed-quality rollouts (checkpoints at 100, 50k, 100k steps) and a non-language baseline. Old LeRobot layout; env needs rebuilding. | **Yes:** `~/Dot_policy/outputs/train/2025-10-10/16-36-17_pusht_resnet_transformer/` |
| [lerobot/pi05_libero_finetuned](https://huggingface.co/lerobot/pi05_libero_finetuned) | π0.5 fine-tuned on LIBERO | 7.47 GB | Optional second baseline A on LIBERO (see "Optional second check" below). Inference should fit 24 GB; full fine-tuning won't. | No |
| [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base) | π0.5 base | 14.47 GB | Not planned: can't be fully fine-tuned in 24 GB | No |

## Base model alternatives (not chosen)

| Option | Size | Status | Why not |
|---|---|---|---|
| π0.7 | ~5B | Not released (no code or weights) | Unavailable |
| π0.5 (LeRobot `pi05`) | ~3.3B | Code in your clone | Not the main policy: full fine-tuning doesn't fit 24 GB, and the config has no freeze options. An earlier PushT run ran out of memory on a 12 GB card. Kept as the optional second check below. |
| EVO1 (LeRobot v0.6.0) | 0.77B | Needs a newer LeRobot than your Oct 2025 clone | Larger than SmolVLA, no clear gain |
| Multitask DiT (LeRobot v0.6.0) | ~450M | Needs newer LeRobot | CLIP-conditioned, not a VLA |
| [EdgeVLA-Tiny](https://huggingface.co/enfuse/edgevla-tiny-fmb) | 164M | Community | Trained on one dataset, offline metrics only |
| SmolVLA with a SmolVLM2-256M backbone | ~0.3B | Config change (`vlm_model_name`) | Loses the pretrained SmolVLA weights; untested |

No SmolVLA successor (a "SmolVLA 2") exists as of this date. Hugging Face models named "smolvla2" (e.g. `Autobrik/smolvla2`, `YIF99/my_smolvla2`) are personal fine-tunes of SmolVLA, not a new version. Don't confuse it with **SmolVLM2**, the vision-language model *inside* SmolVLA (its backbone), which has no action output.

## Optional second check: π0.5

π0.5 is the closest relative of π0.7 you can run: same model family and the same adaptive-RMSNorm action expert. [lerobot/pi05_libero_finetuned](https://huggingface.co/lerobot/pi05_libero_finetuned) (7.5 GB) could serve as a second baseline A. Run it only if SmolVLA shows a gain from subgoals, to see whether the gain holds on a model closer to the paper.

**Costs:**

- **No freeze option.** LeRobot's π0.5 config can't freeze the backbone, so it needs a small code change to train only the action expert. That should fit 24 GB in bf16, but it hasn't been tested.
- **Slower runs,** for both training and evaluation.
- **Less room to improve,** because it scores higher on LIBERO than SmolVLA does.

| | A (baseline) | B (π0.7-style) |
|---|---|---|
| Optional π0.5 check | `lerobot/pi05_libero_finetuned` | π0.5 with the action expert fine-tuned on subgoals |

## Backbones

| Model | Repo size | Note |
|---|---|---|
| [HuggingFaceTB/SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) | 7.92 GB (many formats) | SmolVLA's backbone. When fine-tuning from a SmolVLA checkpoint, the weights come from the checkpoint; only config and tokenizer files are fetched from here. Only a stub is in your cache. |
| [HuggingFaceTB/SmolVLM2-256M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct) | 4.06 GB (many formats) | For the smaller backbone swap |

## World model options

| Candidate | Size | Repo size | Fits 24 GB? | Fine-tune locally? | Role here | On disk |
|---|---|---|---|---|---|---|
| Small diffusion UNet (from scratch) | ~30–100M | – | Easily | Yes | **v1 world model for PushT** | To be written |
| [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo) + GENIMA InstructPix2Pix script | ~1B | 12.96 GB (several variants) | Easily | Yes; script at `~/genima/diffusion/train_instruct_pix2pix_genima.py` | Upgrade for PushT; first choice for LIBERO | Script only |
| [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix) | ~1B | 31.85 GB (several variants) | Easily | Yes | Alternative base for the same script | No |
| [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | 4B | 23.74 GB | Yes, ~13 GB inference | Yes, with LoRA | Strongest trainable option (Apache 2.0) | No |
| [DFloat11/BAGEL-7B-MoT-DF11](https://huggingface.co/DFloat11/BAGEL-7B-MoT-DF11) | 14B | 20.23 GB | Measured: 22.75–23.3 GB peak fully on GPU (55–69 s per image at 512 px); 3.4 GB with all blocks offloaded to CPU (141 s per image) | No, inference only | Zero-shot reference: public BAGEL base compressed losslessly; not the robot-finetuned model. **Tested zero-shot 2026-10-28: 0/3 on LIBERO and 0/5 on π0.7's own website examples.** It draws the named object instead of the action, doesn't make the physical change, and invents rooms. Not usable as our world model without fine-tuning. | **Deleted 2026-10-28** (weights, env and code) to free space. Re-download: `hf download DFloat11/BAGEL-7B-MoT-DF11`. Code changes saved in `artifacts/bagel_subgoal_probe/runtime_changes.patch`. Results kept: `artifacts/bagel_subgoal_probe/`, `artifacts/pi07_reference_subgoals/`. |
| [ByteDance-Seed/BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) | 14B | 29.56 GB | Only with CPU offload | No | The paper's exact starting model | No |
| BAGEL FP8 (community) | 14B | – | ~16 GB with offload | No | Unofficial | No |
| FLUX 3 Image Klein (~Oct 15, 2026) | 5.6B | – | Claimed 8 GB | Unclear | Too new to rely on | No |
| Wan2.1-I2V-14B-480P | 14B | 84 GB | Heavy offload | No | Zero-shot video world model | **Yes:** `~/.cache/huggingface/hub` |
| Wan2.2-TI2V-5B | 5B | – | 24 GB minimum | Tight (~30 GB LoRA) | Predicts whole clips | No |
| Cosmos-Predict2.5 2B | 2B | – | No | No (80 GB for LoRA) | Out of reach | No |
| HunyuanDiT v1.1 distilled | ~1.5B | 14 GB | Yes | – | Not usable: text-to-image only | Yes, `/media/skr/storage/hf_home` |

### Robot-trained world models (existence and metadata checked 2026-10-28)

Unlike everything in the table above, these were already trained on robot data. Only ImageWAM has been tested here so far.

| Candidate | Already trained on | Repo size | License | Fit for our experiment | On disk |
|---|---|---|---|---|---|
| **[ImageWAM FLUX.2 4B LIBERO](https://huggingface.co/yuyangalin/ImageWAM-FLUX.2-4B-LIBERO)** | All four LIBERO suites | 9.04 GB | MIT | **Usual-continuation candidate for LIBERO-10; familiar-instruction candidate for LIBERO-Goal (2026-10-29).** Predicts 16 retained dataset transitions, nominally 0.8 s at the training dataset's 20 fps. Opposing instructions did not reliably select different outcomes in the six-state LIBERO-10 probe. Drawer/pudding chains remained plausible with initial state held fixed; microwave failed. 1.39 s per image, 9.51 GB peak PyTorch allocation after unloading Qwen3. [Controlled follow-up](artifacts/imagewam_prompt_control_probe/README.md). From three LIBERO-Goal start scenes, familiar instructions yield 63/90 task matches after 1 call and 59/60 after 5 calls, versus real-frame reference scores of 37/50 and 47/50. These are pixel-based task classifications, not success rates or ceilings. ([Language test and review](artifacts/imagewam_goal_language_probe/README.md).) At LIBERO-10 subtask boundaries with our subtask wording (18 boundaries, 9 tasks) the instruction did **not** select the subtask: the model performs the scene's habitual step, and an already-done instruction makes it stall ([boundary test](artifacts/imagewam_boundary_probe/README.md)). Closed-loop benefit remains untested. | Yes |
| **[Cosmos Policy LIBERO 2B](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B)** | LIBERO | 3.95 GB | Not stated in the repo metadata; check the model card | Strong alternative. Produces future camera images alongside actions and state predictions. See the [Cosmos report](cosmos_policy_report.md) for text encoder options, budget and pending checks. | No |
| **[SuSIE](https://github.com/kvablack/susie#model-weights)** ([weights](https://huggingface.co/kvablack/susie)) | BridgeData + Something-Something | 3.44 GB (Flax UNet; also needs the SD 1.5 VAE and text encoder) | MIT | Direct image-and-language subgoal interface, but the released checkpoint's transfer to LIBERO needs testing. Code cloned at `../susie`. | Code only |
| **[BFL FLUX 3 Action](https://github.com/black-forest-labs/flux-action)** | Released DROID and SO-101 variants | – (7B world-action model) | Code Apache-2.0; weight license not checked | Already robot-trained, but these releases target different data and observation conventions from LIBERO. | No |

## Local subgoal checkpoint check (2026-10-28)

**SuSIE uses a separate, robot-trained checkpoint:** `kvablack/susie`. The cloned [README](../susie/README.md) documents a UNet trained on BridgeData and Something-Something for 40,000 steps. Its inference helper loads this UNet with the Stable Diffusion 1.5 VAE and text encoder (`runwayml/stable-diffusion-v1-5:flax`). The training configuration initializes from `kvablack/instruct-pix2pix-flax`. Cloning the repository did not download these weights.

**The downloaded base BAGEL checkpoint ran**, using [probe_bagel.py](probe_bagel.py), the local `bagel_dfloat11` checkout and `.venv-bagel` (all deleted 2026-10-28; the code changes are in `artifacts/bagel_subgoal_probe/runtime_changes.patch`). This is the public BAGEL base, not the robot-finetuned world model described in the π0.7 paper's Appendix C.

Three selected LIBERO-10 frames were tested with one subtask prompt and seed each, at 512×512 with 50 sampling timesteps. [View current/generated/reference images](artifacts/bagel_subgoal_probe/results/comparison.png).

| Case | Visual observation |
|---|---|
| Close drawer | Drawer remains open; robot geometry is broken and a nearby bottle disappears. |
| Close microwave | Door remains open; most of the robot and an unrelated mug disappear. |
| Place pudding beside plate | Invents an unboxed dessert and an extra plate instead of moving the existing pudding package. |

These three outputs do not support using base BAGEL directly as the subgoal generator for this run. This is a qualitative probe, not a measured success rate or a policy evaluation. References are later subtask-end frames and were never generation inputs; task-relative “right” in the pudding instruction differs from image-right.

Runtime: about 141–143 seconds per image on the RTX PRO 4000 Blackwell with all compressed blocks CPU-offloaded, and about 3.15 GiB peak PyTorch GPU allocation. Other GPU jobs were running, so these are not optimized latency figures. PyTorch SDPA substitutes for FlashAttention; packed causal/noncausal attention and grouped-query cases passed a numerical comparison against explicit attention, but full-model equivalence to the upstream kernel was not tested. [Settings and timings](artifacts/bagel_subgoal_probe/results/report.json), [visual assessment](artifacts/bagel_subgoal_probe/results/assessment.json), [environment](artifacts/bagel_subgoal_probe/environment.json), and [runtime patch](artifacts/bagel_subgoal_probe/runtime_changes.patch) are saved with the images.

Re-run from `/home/skr/Downloads/pi_07`: `.venv-bagel/bin/python -u pi07_code/probe_bagel.py`.

## The paper's own models (reference)

| Model | Role in π0.7 | Availability |
|---|---|---|
| Gemma3 4B | Policy backbone (π0.7 is initialized from it, not from π0.6 weights) | Gated on Hugging Face (accept Google's license); about 8.6 GB in bf16 (approximate) |
| BAGEL-7B-MoT | World model, fine-tuned | Base checkpoint is public; the downloaded weights are the base, not the paper's robot-finetuned weights |
| π0, π0-FAST, π0.5 | Earlier models | Public in [openpi](https://github.com/Physical-Intelligence/openpi) and LeRobot |
| π0.6, π0.7 | Earlier / this model | Not released. A hosted Pi API beta (Oct 2026) offers fine-tuning and inference by application. Which models it covers is unclear. |

## Sources

- Hugging Face model pages linked in the tables (sizes, downloads and dates read via the Hugging Face API on 2026-10-28)
- [LeRobot v0.6.0 release blog](https://huggingface.co/blog/lerobot-release-v060)
- [Bagel-DFloat11 (GitHub)](https://github.com/LeanModels/Bagel-DFloat11)
- [FLUX 3 Image Klein report (zeli.app)](https://zeli.app/story/50123580)
- [AMD ROCm: Wan2.2 fine-tuning on a single GPU](https://rocm.blogs.amd.com/artificial-intelligence/finetuning-wan-part1/README.html)
- [Fine-tuning Cosmos Predict 2.5 with LoRA (mirror of HF blog)](https://huggingface-proxy-6ci.pages.dev/blog/nvidia/cosmos-fine-tuning-for-robot-video-generation)
- [The AI Insider: Physical Intelligence opens Pi API beta](https://theaiinsider.tech/2026/10/14/physical-intelligence-opens-pi-api-beta-for-robot-model-training-and-deployment/)
