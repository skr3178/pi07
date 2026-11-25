# Toy π0.7 on PushT: reconstruction plan

As of 2026-10-28. Source paper: [pi_07.pdf](pi_07.pdf) (π0.7, Physical Intelligence, April 2026). Equations: [equations.md](equations.md). No official π0.7 code or weights exist, so the paper is the spec.

The toy runs π0.7's test-time loop on PushT. A SmolVLA policy is fine-tuned with π0.7's richer prompt. A small diffusion world model draws intermediate goal images, and those images steer the policy toward completing the push. The main test is whether that policy (B) beats the same SmolVLA fine-tuned on PushT without subgoals (A).

## Scope

**Claims the toy tests:**

- Intermediate goal images from a diffusion world model raise PushT success over the same policy without them (A vs B).
- Randomly dropping prompt parts in training lets one model run with or without subgoals.
- Add-on: with episode metadata in the prompt, mixed-quality rollouts improve the policy instead of hurting it (paper Fig. 18 left).

**Out of scope on PushT:** language following and composing new tasks (PushT has one task with a fixed target; LIBERO later), cross-embodiment transfer, real robots, and web or human-video data.

**Hardware:** 1× NVIDIA RTX PRO 4000 Blackwell (24 GB), 61 GB RAM.

## Side by side: run-time loop

**Paper (π0.7)**

```
 task text ---------------------------+
 cameras (up to 4 views) + robot state+
                                      |
                                      v
                    +-----------------------------+
                    | HIGH-LEVEL POLICY           |   pi0.7 fine-tuned on
                    | -> next subtask text        |   coaching data, or a
                    +--------------+--------------+   human typing steps
                                   | subtask
                                   v
                    +-----------------------------+
 desired metadata ->| WORLD MODEL (BAGEL 14B)     |   flow-matching image
 (Quality 5,        | images + subtask + metadata |   generator; refreshed on
  Mistake false,    | -> subgoal images (<= 3)    |   a new subtask or every 4 s
  fast Speed)       +--------------+--------------+
                                   | subgoal images
                                   v
                    +-----------------------------+
 task, subtask,  -->| pi0.7 VLA (~5B)             |
 metadata,          |                             |
 control mode,      |                             |
 camera history,    |                             |
 robot state        |                             |
                    +--------------+--------------+
                                   | 50-step action chunk
                                   v
                    execute 15 or 25 steps; the next chunk is
                    computed while these run (real-time chunking)
                                   |
                                   v
                    PD controller (+ IK) --> robot @ 50 Hz
```

**Ours (PushT)**

```
 PushT env (10 fps; 96x96 image + 2-D agent position)
        |
        v
 +---------------------------+   new subgoal every 40 steps (4 s),
 | SUBTASK                   |   or when the subtask changes
 | v1: none (fixed horizon)  |------------------------+
 | v2: phase from T pose:    |                        |
 |  approach/push/rotate/    |                        v
 |  align                    |          +------------------------------+
 +---------------------------+          | DIFFUSION WORLD MODEL        |
                                        | current frame (+ subtask     |
 metadata (add-on):    ---------------->|  + metadata)                 |
 Quality 5, Mistake false,              | -> intermediate goal image   |
 fast Speed                             | (96x96, small UNet)          |
                                        +--------------+---------------+
                                                       | subgoal image
                                                       v
                    +-----------------------------+
 task text,      -->| SmolVLA (~450M)             |
 subtask (v2),      | fine-tuned with the         |
 metadata (add-on), | pi0.7 prompt                |
 agent position     |                             |
                    +--------------+--------------+
                                   | 16-step chunk of 2-D target positions
                                   v
                    execute 8 steps (synchronous, no RTC)
                                   |
                                   v
                    PushT env @ 10 fps --> repeat
```

## Side by side: inside the policy

**Paper (π0.7 VLA)**

```
 PREFIX: Gemma3 4B backbone (400M vision encoder shared by all images)

 +------------------+------------------+--------------------------+-------------+
 | OBSERVATION      | SUBGOAL          | TEXT                     | STATE       |
 | <=4 cameras x    | <=3 views        | Task: ... Subtask: ...   | each history|
 | <=6 frames, 1 s  | (no rear cam),   | Speed: ... Quality: ...  | state = one |
 | apart, 448x448;  | same encoder     | Mistake: ...             | token via a |
 | MEM compresses   |                  | Control Mode: ...        | linear proj |
 | the history to   |                  |                          |             |
 | 1 frame's tokens |                  |                          |             |
 +------------------+------------------+--------------------------+-------------+
  attends to:        itself + the        causal (all earlier       (not stated
  itself only        observation         tokens)                    in paper)
                                   |
                                   | keys/values from every backbone layer
                                   v
 SUFFIX: action expert (860M transformer, flow matching)
 +-----------------------------------------------------------------------------+
 | 50 noisy action tokens: attend to each other + the whole prefix;            |
 | flow-matching time step injected via adaptive RMSNorm                       |
 +-----------------------------------------------------------------------------+
                                   |
        5 denoising steps; CFG on metadata, beta in {1.3, 1.7, 2.2}
                                   v
                    action chunk: 50 steps (joint or end-effector)
```

Note: the paper's Fig. 19 caption places subgoal images after the text prompt, while Sec. VI-B and the example prompt place them before it. This diagram follows the main text.

**Ours (SmolVLA)**

```
 PREFIX: SmolVLM2-500M backbone, first 16 layers (frozen, incl. vision encoder)

 +------------------+------------------+--------------------------+-------------+
 | OBSERVATION      | SUBGOAL          | TEXT                     | STATE       |
 | 1 top-down image | 1 image from the | Task: push the T onto    | 2-D agent   |
 | 96x96, upscaled  | world model,     | the target. Subtask: ... | position,   |
 | to SmolVLM size; | same encoder     | (v2) Quality: ...        | one token   |
 | no history       |                  | Mistake: ... (add-on)    | (linear)    |
 +------------------+------------------+--------------------------+-------------+
  attends to:        <------- one block: observation + subgoal + text ------->
  itself only        (the subgoal block starts with a 1 in att_masks,        sees all
                      so the observation cannot see the goal)                 earlier
                                   |
                                   v
 SUFFIX: SmolVLA action expert (width 0.75x the VLM, flow matching)
 +-----------------------------------------------------------------------------+
 | 16 noisy action tokens (2-D target positions), attend to the prefix         |
 +-----------------------------------------------------------------------------+
                                   |
        5-10 denoising steps; CFG on metadata (add-on)
                                   v
                    action chunk: 16 steps, execute 8
```

## Side by side: training

**Paper (π0.7)**

```
 LOSSES
   backbone      <-- cross-entropy on FAST action tokens
   action expert <-- flow-matching loss; its gradients stop before the
                     backbone (knowledge insulation)
   world model   <-- trained separately (fine-tuned from BAGEL): flow
                     matching toward the end-of-segment frame

 PER-EXAMPLE RANDOMIZATION
   subgoals present in 25% of examples
     real frame: end of segment (p=0.25) or 0-4 s ahead (p=0.75)
     plus extra examples using world-model images instead
   subtask text dropped 30% of the time when a subgoal is present
   metadata dropped 15% (whole) + 5% per field
   history frames dropped 30%; rear camera dropped 30%
   simulated inference delay of 0-12 steps (training-time RTC)
```

**Ours, PushT (optional debugging variant)**

```
 LOSSES
   policy (SmolVLA) <-- flow-matching loss on 16-step chunks; VLM frozen,
                        so only the expert and projections train
   world model      <-- trained separately, from scratch: diffusion loss
                        toward the frame 20 steps ahead (v1) or the end of
                        the current phase (v2)

 PER-EXAMPLE RANDOMIZATION (paper's rates, horizons in steps at 10 fps)
   subgoals present in 25% of examples
     real frame: the world model's target frame (p=0.25)
                 or 0-40 steps ahead (p=0.75)
     plus extra examples using world-model images, once it exists
   subtask text dropped 30% when a subgoal is present (v2)
   metadata dropped 15% (whole) + 5% per field (add-on)
   no history, rear camera or RTC to drop
```

**Ours, LIBERO (main experiment)**

Same layout as the paper's block. Lines starting `#` say what changed and why; a line marked `# same` copies the paper.

```
 STARTING WEIGHTS                    (the paper's block doesn't list these;
                                      pi0.7 starts from Gemma3 4B)
   policy      <-- lerobot/smolvla_libero: SmolVLA ~450M, already fine-tuned
                   on lerobot/libero
                   # why: ~500 demos plus rollouts are far too few to train
                   #   a VLA from a raw backbone; 450M fits 24 GB with room
                   #   to spare; an official LIBERO checkpoint exists
   world model <-- InstructPix2Pix (sd-turbo base, GENIMA script);
                   FLUX.2 Klein 4B + LoRA if that's too weak
                   # why: BAGEL 14B can't be fine-tuned on 24 GB. Same idea
                   #   as the paper: a pretrained image editor fine-tuned
                   #   on robot frames (as SuSIE does)

 DATA                                (the paper: demos on many robots,
                                      failures, pi0.6* RL rollouts, web
                                      and human video)
   lerobot/libero_10_subtask: 500 demos, 10 tasks, subtask ids mapped to
     text (dataset.md); cameras image/image2 -> camera1/camera2
   + new simulator rollouts on the same tasks: successful, slow, failed
     and recovered, from smolvla_libero with perturbations
                   # why: the released demos are all high quality.
                   #   Metadata needs a mixed-quality set; the paper got
                   #   one from failures and earlier models' rollouts
   no web or human-video data
                   # why: nothing in a single-arm sim test would use it

 LOSSES
   policy (SmolVLA) <-- flow-matching loss on 50-step chunks;
                        backbone frozen, so only the action expert and
                        projections train
                        # why: replaces the FAST-token loss + knowledge
                        #   insulation. A frozen backbone gets no expert
                        #   gradients either, and needs no extra loss head.
                        #   Chunk stays 50 to match the checkpoint.
                        # note: smolvla_libero was trained with the
                        #   backbone unfrozen; unfreeze if the expert can't
                        #   use the subgoal/metadata tokens
   world model      <-- trained separately: diffusion loss toward the
                        end-of-segment frame (segment = one subtask id's
                        span), scene camera only
                        # same target as the paper
                        # why scene camera only: the paper predicts up to
                        #   3 views; one view halves the world model's cost.
                        #   Add the wrist view later if needed

 PER-EXAMPLE RANDOMIZATION (paper's rates; 10 fps, so 4 s = 40 steps)
   subgoals present in 25% of examples              # same
     real frame: end of segment (p=0.25)            # same
                 or 0-40 steps ahead (p=0.75)       # same, in steps
     plus extra examples using world-model images   # same, once trained
   subtask text dropped 30% when a subgoal is present           # same
   metadata dropped 15% (whole) + 5% per field                  # same
     demos without validated labels get an explicit "unknown",
     never a guessed value
                        # why: dropout must stay distinguishable from
                        #   missing labels (dataset.md, row E)
   history frames: none to drop
                        # why: SmolVLA has no MEM-style history. LIBERO-10
                        #   demos run 18-41 s (median per task), short
                        #   enough without it
   rear camera: none to drop
                        # why: LIBERO has only scene + wrist cameras
   training-time RTC: skipped
                        # why: the sim runs synchronously, so there is no
                        #   inference delay to hide
```

Before training, drop task 3's 1-frame id-15 segments, and leave task 9 (one segment, no split) out of the subtask comparisons. See dataset.md.

## Side by side: every component

| Component | Paper (π0.7) | Our replacement (PushT toy) | Why |
|---|---|---|---|
| **Policy model** | | | |
| Base model | ~5B VLA on the π0.6-MEM design (Sec. IV) | LeRobot SmolVLA, ~450M | π0.5 (3.3B) can't be fully fine-tuned in 24 GB; an earlier π0.5 PushT run ran out of memory on a 12 GB card |
| Starting weights | Backbone initialized from Gemma3 4B, not from π0.6 weights; π0.6 influences it only through rollout data | Fine-tune a SmolVLA checkpoint: `smolvla_base`, or `naonaon/smolvla_pusht` if it evaluates well | 206 demos are far too few to train a VLA from a raw backbone |
| Backbone | Gemma3 4B incl. 400M vision encoder | SmolVLM2-500M, first 16 layers, frozen | Same role, fraction of the size |
| Action expert | 860M, flow matching, adaptive RMSNorm | SmolVLA's expert (width 0.75× the VLM), flow matching | Same mechanism |
| Chunk / executed steps | 50 / 15 or 25 at 50 Hz | 16 / 8 at 10 fps | 50 steps would be 5 s without re-planning; Diffusion Policy uses 16 / 8 on PushT |
| Denoising steps | 5 | 5–10 | Same idea |
| Image input | Up to 4 cameras at 448×448 | 1 top-down 96×96 image | That's what PushT provides |
| Video history | MEM: 6 frames, 1 s apart, compressed | Skipped | Short task |
| State input | Linear projection, one token per history state | SmolVLA's linear projection of the 2-D agent position | Same design |
| Knowledge insulation | FAST-token loss on the backbone, expert gradients stopped | Frozen VLM | No expert gradients reach the backbone either |
| Attention mask | Observation ↔ itself; subgoal sees observation; text causal | Subgoal block that sees the observation; text bidirectional | Keeps the key property with a one-line change |
| Real-time chunking | Training-time RTC, 0–12 step delays | Skipped | Sim runs synchronously |
| **Prompt** | | | |
| Task text | Short description | "push the T onto the target" (one fixed task) | PushT has one task |
| Subtask text | Human-annotated segments | v1: none. v2: phases from the T's pose (approach / push / rotate / align) | The image dataset has no T pose, so v2 needs extra work |
| Subgoal images | Up to 3 views, near-future scene | 1 image, 96×96 | One camera |
| Speed metadata | Episode length, 500-step bins | Episode length (max 300 steps), e.g. 25-step bins (add-on) | Same signal, rescaled |
| Quality metadata | Human score 1–5 | Define after validating reward/success provenance; log raw coverage on new rollouts (add-on) | The inspected dataset has no true success labels; recorded reward must not be treated as raw coverage |
| Mistake metadata | Human per-segment flag | Known perturbation intervals in new rollouts, with a validated heuristic if needed (add-on) | A temporary coverage drop can be a useful rotation or repositioning, rather than a mistake |
| Control mode | `joint` / `ee` tag | Skipped | One action space |
| Prompt dropout | Subgoal 25%; subtask −30% with subgoal; metadata −15% whole, −5% per field | Same rates | Part of what's being tested |
| CFG | On metadata, β ∈ {1.3, 1.7, 2.2} | Same, once metadata is added | Same |
| **World model** | | | |
| Model | BAGEL-7B-MoT (14B total), fine-tuned | Small diffusion UNet, trained from scratch at 96×96 | PushT frames are tiny and simple; BAGEL can't be fine-tuned on 24 GB |
| Inputs | Current images + subtask + metadata | Current frame (+ subtask in v2, + metadata as an add-on) | Same conditioning, staged |
| Target | End-of-segment frame | v1: frame 20 steps ahead (as SuSIE does). v2: end of the current phase. | v1 needs no labels |
| Training data | High-quality robot segments, human video, image-editing and video datasets | (frame, future frame) pairs from the 206 demos | In-domain pairs are plentiful |
| Refresh at run time | New subtask or every Δ = 4 s, async | Every 40 steps (4 s) or new phase, synchronous | Same rule |
| **High-level policy** | π0.7 fine-tuned on coaching data | v1: none. v2: rule-based phase detector | Single task |
| **Data** | | | |
| Demonstrations | Teleoperation on many robots | 206 human PushT demos (`lerobot/pusht`) | The dataset |
| Mixed-quality / autonomous data | Failures, interventions, π0.6* RL rollouts | Rollouts of your DOT PushT checkpoints (100, 50k, 100k steps) plus action noise (add-on) | Same "learn from earlier models" idea |
| Non-robot data | Human video, web data | Skipped | Can't be tested on PushT |
| Environment | Mobile and static bimanual robots, UR5e | PushT simulator (`gym-pusht`), 10 fps, max 300 steps | Simulation only |
| **Evaluation** | | | |
| Main comparison | π0.7 vs π0.7 with generated subgoals (GC) | A (no subgoals) vs B (world-model subgoals), same seeds | See Experiments |
| Mixed-quality scaling | Fig. 18 left: top 30/50/80/100%, with and without metadata | Same grid on demos + DOT rollouts (add-on) | Same |
| Language, composition, cross-embodiment | Sec. IX-B to IX-D | Out of scope; LIBERO later | PushT has one task |

## Experiments

**Main: A vs B (closed loop)**

| | A: Baseline | B: π0.7-style |
|---|---|---|
| Starting checkpoint | Same | Same |
| Training data | 206 PushT demos | Same demos, plus subgoal images (later subtask and metadata), paper's dropout rates |
| At test time | Current image + task text | Same, plus a world-model subgoal refreshed every ~40 steps |
| Everything else | Same steps, chunk 16, execute 8 | Same |

Run both on the same 100 seeded PushT episodes. Compare success rate (≥95% coverage), average maximum coverage and episode length. Only the subgoals differ, so a gap in B's favor comes from them.

**Offline diagnostic.** On held-out demos, compare B's action error given (a) the real future frame, (b) the world model's image, (c) no subgoal. Real future frames don't exist in a live rollout, so this best case is only measurable offline. It shows whether a shortfall comes from the world model or the policy.

**Sanity check.** Run B with no subgoal. Thanks to dropout it should still roughly match A.

**Add-on: metadata.** Repeat paper Fig. 18 left on demos plus DOT rollouts: train on the top 30/50/80/100% by coverage, with and without metadata (8 runs), then sweep the CFG weight.

## Checks before training

**Dataset format audit (completed 2026-10-28).** Inspected the actual `lerobot/pusht` files at revision `7628202a2180972f291ba1bc6723834921e72c19`, saved under `artifacts/pusht_preflight/dataset`. The local LeRobot clone and dataset both use v3.0. The dataset contains 206 episodes, 25,650 rows, 96×96 RGB video at 10 fps, one task string, and finite 2-D states/actions. Episode bounds, frame indices, timestamps, video offsets and frame counts passed the audit. It supports 22,560 complete 16-action windows, 21,530 unclipped t+20 pairs and 17,410 unclipped t+40 pairs without crossing episode boundaries. These windows are correlated samples from 206 trajectories, not independent demonstrations.

- Reproduce: `python3 pi07_code/inspect_pusht.py pi07_code/artifacts/pusht_preflight/dataset --output pi07_code/artifacts/pusht_preflight` from the workspace root.
- [Machine-readable report](artifacts/pusht_preflight/dataset_report.json) and [sample current/future frames](artifacts/pusht_preflight/future_frame_pairs.png). These are real dataset frames, not generated subgoals.
- **Label issue:** every stored `next.success` is false, and the largest recorded `next.reward` is 0.9488797. This does not establish that all demonstrations failed. Audit conversion/label semantics before deriving quality bins or calling episode length time-to-success. Raw coverage is not stored explicitly.
- No block pose, subtask, segment boundary, quality, speed-bin or mistake annotation is stored. Fixed-horizon goals need no phase labels. Split by episode before fitting a world model; a policy checkpoint already trained on these demos invalidates a new claim that its held-out subset was unseen.
- The available OpenCV decoder failed on the AV1 video; FFmpeg with `libdav1d` decoded the sampled frames. A clean LeRobot runtime still needs a real processor/forward-pass check: camera-key mapping, RGB ordering, normalization, action padding/trimming, and episode-end action masks. The audit does not certify policy inference.

**Released subgoal checkpoint check (required before choosing the world model).** The official SuSIE code is cloned at `../susie`. Its released [checkpoint](https://huggingface.co/kvablack/susie) is a roughly 3.44 GB Flax UNet trained on BridgeData and Something-Something, with a separate Stable Diffusion v1.5 VAE and text encoder. The weights and compatible runtime have not been installed here. Cloning the code does not download them. Its sampler expects a 256×256 RGB uint8 image and a text instruction; it does not provide π0.7 quality/speed/mistake conditioning out of the box.

1. Load the released checkpoint in an isolated environment and verify the official robot-image example as a control. Check Blackwell/runtime compatibility before attributing any failure to the model.
2. Run the same checkpoint on the saved PushT samples, resized to 256×256, with a fixed task prompt and several recorded random seeds. Save current image, generated goal and real future frames side by side. Inspect preservation of the T shape, pusher and target, plausible reachable motion, and progress toward alignment; measure latency. This is a zero-shot domain-transfer probe, not proof of PushT competence.
3. If outputs are useful, test them with the goal-conditioned policy. If they are unsuitable, select PushT fine-tuning or the small from-scratch UNet based on that evidence. Poor zero-shot SuSIE output does not disprove the visual-subgoal mechanism. The separately downloaded BAGEL checkpoint is a different model and does not establish SuSIE readiness.

## Build stages

0. **Validate inputs and checkpoints.** Complete the checks above, create the LeRobot env with PushT, and run one real policy batch before a rollout. Download `naonaon/smolvla_pusht` and evaluate it for 50 episodes, executing 8 of its 50 steps. If it works reasonably, it becomes the shared A/B starting point; otherwise fine-tune from `smolvla_base`. Keep pilot seeds separate from the final evaluation seeds.
1. **Baseline A.** Fine-tune SmolVLA on PushT (chunk 16, execute 8). Evaluate on 100 seeded episodes.
2. **Subgoal pairs.** Build (frame t, frame t+20) and (frame t, frame t+0..40) pairs from the demos.
3. **World model.** Use the checkpoint probe to choose a released-model adaptation or the small diffusion UNet at 96×96. Validate held-out geometry, progress and latency before coupling it to the policy; plausible-looking images alone are insufficient.
4. **Policy B.** Add the subgoal image slot, the attention-mask change and the subgoal sampler. Train with real future frames, then mix in generated ones.
5. **Evaluate.** A vs B closed loop, the offline diagnostic, and B without subgoals.
6. **Add-ons.** v2 phase subtasks; metadata with DOT rollouts; BAGEL zero-shot as a curiosity; LIBERO for language.

## Assets already on disk

| Asset | Location | Use |
|---|---|---|
| LeRobot clone (Oct 2025, commit 76a425c6) | `~/Downloads/lerobot` | SmolVLA, PushT env, training and eval scripts |
| Earlier π0.5 PushT scripts | `~/Downloads/lerobot/test_pi05_pusht.py`, `simple_test_pi05.py`, `visualize_pi05_pusht.py` | Dataset loading to reuse |
| DOT PushT policy ([skr3178/dot_resnet](https://github.com/skr3178/dot_resnet)) | `~/Dot_policy/outputs/train/2025-10-10/16-36-17_pusht_resnet_transformer/` (checkpoints at 100, 50k, 100k steps; 93 eval videos) | Mixed-quality rollouts, non-language baseline. Old LeRobot layout; its env needs rebuilding (`pip_list_dot2.txt`). |
| ~~BAGEL DFloat11~~ (deleted 2026-10-28) | – | Tested zero-shot and ruled out (0/3 on LIBERO, 0/5 on π0.7's examples); weights, env and code removed to free space. Results kept in `artifacts/bagel_subgoal_probe/` and `artifacts/pi07_reference_subgoals/`. Re-download: `hf download DFloat11/BAGEL-7B-MoT-DF11`. |
| GENIMA InstructPix2Pix training script | `~/genima/diffusion/train_instruct_pix2pix_genima.py` | World model upgrade if the small UNet is too weak |
| Wan2.1-I2V-14B-480P (84 GB) | `~/.cache/huggingface/hub` | Zero-shot video world model, optional |

## Code touchpoints (LeRobot clone)

| What | Where |
|---|---|
| Prompt builder (subtask, metadata, dropout) | New processor step before `SmolVLANewLineProcessor`, `src/lerobot/policies/smolvla/processor_smolvla.py:107` |
| Subgoal tokens and attention mask | `embed_prefix`, `src/lerobot/policies/smolvla/modeling_smolvla.py:534`. Image tokens at line 575, language at 599, state at 612. Give the subgoal block a leading `1` in `att_masks`. |
| CFG on metadata | `sample_actions` (line 709) and `denoise_step` (line 750) in `modeling_smolvla.py` |
| PushT env settings | `PushtEnv` in `src/lerobot/envs/configs.py`: 10 fps, 300-step episodes, observations configured at 384×384 |
| New code | Subgoal pair builder and sampler; diffusion world model; closed-loop A/B eval script; offline diagnostic |

## Environment setup

- The old `lerobot2` conda env no longer exists.
- The root drive is ~95% full, so the env and caches go on `/media/skr/storage` (~150 GB free).
- The Blackwell GPU needs a CUDA 12.8 build of PyTorch, and LeRobot pins `torch<2.8`, so use 2.7.1.

```bash
conda create -p /media/skr/storage/conda_envs/pi07 python=3.10 -y
conda activate /media/skr/storage/conda_envs/pi07
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
cd ~/Downloads/lerobot && pip install -e ".[smolvla,pusht]"
export HF_HOME=/media/skr/storage/hf_cache
```

## Reference: SmolVLA PushT checkpoints (checked 2026-10-28)

There is no official one; the `lerobot` org has only `diffusion_pusht` and `vqbet_pusht`. About 50 community uploads exist, and none reports a success rate. All use 50-step chunks executed whole.

| Checkpoint | Trained from | Steps | Notes |
|---|---|---|---|
| [naonaon/smolvla_pusht](https://huggingface.co/naonaon/smolvla_pusht) (Dec 2025) | `smolvla_base` | 100k, batch 24 | Best documented: 1 camera, actions cut from 6-D to 2-D, eval command. Newer config format. |
| [n3puiol/smolvla_pusht](https://huggingface.co/n3puiol/smolvla_pusht) (Aug 2025) | Not from `smolvla_base` | 20k | Most downloaded; older format may not load in current LeRobot |
| [crislmfroes/smolvla-pusht](https://huggingface.co/crislmfroes/smolvla-pusht) | Not from `smolvla_base` | 5k | Barely trained |

## Reference: base model alternatives (checked 2026-10-28)

- **No SmolVLA successor exists.** The only SmolVLA is the 450M model from June 2025, plus benchmark fine-tunes from `lerobot` (`smolvla_libero_plus`, `smolvla_robocasa`, `smolvla_vlabench`, …).
- **LeRobot v0.6.0** (July 7, 2026) added **EVO1** (0.77B), the only new VLA under 1B; it needs a newer LeRobot. **Multitask DiT** (~450M) is CLIP-conditioned, not a VLA. Other additions are 2–6B.
- **EdgeVLA-Tiny** (164M, community) is trained on one dataset with offline metrics only.
- **Smaller option:** set SmolVLA's `vlm_model_name` to SmolVLM2-256M. Loses the pretrained SmolVLA weights; untested.
- **Decision:** stay with SmolVLA 450M.

## Reference: world model options (checked 2026-10-28)

There is no smaller official BAGEL. The paper's "14B" is the total size of BAGEL-7B-MoT, ByteDance's only release.

| Candidate | Size | Fits 24 GB? | Fine-tune locally? | Role here |
|---|---|---|---|---|
| **Small diffusion UNet (from scratch)** | ~30–100M | Easily | Yes | **v1 world model for PushT** |
| InstructPix2Pix / sd-turbo via GENIMA | ~1B | Easily | Yes; script on disk | Upgrade if the UNet is too weak |
| FLUX.2 Klein 4B | 4B | Yes, ~13 GB inference | Yes, with LoRA | Strongest trainable option; Apache 2.0 |
| BAGEL DFloat11 (downloaded) | 14B | Just: 21.8 GB peak | No, inference only | Zero-shot reference |
| FLUX 3 Image Klein (~Oct 15, 2026) | 5.6B | Claimed 8 GB | Unclear | Too new |
| Wan2.2-TI2V-5B | 5B | 24 GB minimum | Tight (~30 GB LoRA) | Predicts whole clips |
| Cosmos-Predict2.5 2B | 2B | No | No (80 GB for LoRA) | Out of reach |

## Risks and open questions

- **No language test.** PushT has one task with a fixed target, so instruction following waits for LIBERO.
- **Community checkpoints are untested.** Their 50-step open-loop chunks may react too slowly for PushT; stage 0 settles this.
- **Image size.** The dataset is 96×96, but LeRobot's PushT env config renders 384×384. Check what reaches the policy before evaluating.
- **World model quality.** A blurry or wrong T pose in the subgoal could mislead the policy. Mixing generated subgoals into training, as the paper does, should help; the offline diagnostic measures it.
- **Over-reliance on subgoals.** With a goal frame the task becomes almost pure inverse dynamics. The paper limits subgoals to 25% of examples for this reason; keep that rate.
- **v2 labels.** Phase subtasks need the T's pose, which `lerobot/pusht` doesn't store. Options: the keypoint version of the dataset, detecting the T in the image, or logging it during new rollouts.
- **Frozen backbone.** The expert may struggle to use subgoal tokens through a frozen VLM. Fallback: unfreeze it (`train_expert_only=False`).
- **Small dataset.** 206 demos invite overfitting, for the world model especially. Watch held-out samples.

## Sources

Official π0.7 code (checked 2026-10-28: none released)
- [π0.7 project page](https://pi.website/pi07)
- [Physical-Intelligence/openpi (GitHub)](https://github.com/Physical-Intelligence/openpi): π0, π0-FAST, π0.5 only
- [LeRobot policies folder (GitHub)](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies): no π0.6 or π0.7
- [The AI Insider: Physical Intelligence opens Pi API beta](https://theaiinsider.tech/2026/10/14/physical-intelligence-opens-pi-api-beta-for-robot-model-training-and-deployment/)

PushT
- [lerobot/pusht dataset](https://huggingface.co/datasets/lerobot/pusht): 206 episodes, 25,650 frames, 10 fps, 96×96 images, 2-D state and action
- [naonaon/smolvla_pusht](https://huggingface.co/naonaon/smolvla_pusht), [n3puiol/smolvla_pusht](https://huggingface.co/n3puiol/smolvla_pusht), [crislmfroes/smolvla-pusht](https://huggingface.co/crislmfroes/smolvla-pusht)

Base model alternatives
- [LeRobot v0.6.0 release blog](https://huggingface.co/blog/lerobot-release-v060)
- [lerobot models on Hugging Face](https://huggingface.co/lerobot)
- [EdgeVLA-Tiny model card](https://huggingface.co/enfuse/edgevla-tiny-fmb)

World model options
- [FLUX.2 Klein 4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- [FLUX 3 Image Klein report (zeli.app)](https://zeli.app/story/50123580)
- [Bagel-DFloat11 (GitHub)](https://github.com/LeanModels/Bagel-DFloat11)
- [DFloat11/BAGEL-7B-MoT-DF11 (Hugging Face)](https://huggingface.co/DFloat11/BAGEL-7B-MoT-DF11)
- [NVIDIA Megatron Bridge BAGEL docs](https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/bagel/bagel.html)
- [AMD ROCm: Wan2.2 fine-tuning on a single GPU](https://rocm.blogs.amd.com/artificial-intelligence/finetuning-wan-part1/README.html)
- [Fine-tuning Cosmos Predict 2.5 with LoRA (mirror of HF blog)](https://huggingface-proxy-6ci.pages.dev/blog/nvidia/cosmos-fine-tuning-for-robot-video-generation)
