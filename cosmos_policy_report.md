# Cosmos Policy LIBERO: findings and proposed use

Research date: 2026-10-28. Companion to [model.md](model.md) and [dataset.md](dataset.md).

**Conclusion:** Cosmos Policy LIBERO 2B is a plausible pretrained source of future camera images for our SmolVLA experiment. The full 45 GB T5 download is avoidable: standard LIBERO instructions already have cached embeddings, and custom instructions can potentially use an encoder-only release of the same T5 model. Local image quality, runtime, memory use and the alternative encoder-loading path remain untested.

## What the checkpoint does

[NVIDIA's Cosmos Policy LIBERO checkpoint](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B) is a robot-trained model built from Cosmos Predict2 2B. It was trained on the four LIBERO suites, including LIBERO-10. It jointly predicts actions, future robot state, future camera images and a value estimate.

The text encoder converts a supplied instruction into features. It does **not** generate instructions or choose the next subtask. Future images come from the Cosmos Policy model. [Official inference example](https://github.com/NVlabs/cosmos-policy#quick-start).

| Interface | LIBERO configuration |
|---|---|
| Visual input | Current external and wrist camera views; processed at 224 × 224 |
| Robot state | 9 values: 2 gripper joint positions, 3 end-effector position values, 4 quaternion values, in that order |
| Language | Task instruction encoded with `google-t5/t5-11b`, or its cached embedding |
| Actions | Chunk of 16 actions, each with 7 values |
| Future observations | External and wrist images and proprioception at the action-chunk horizon |

The official evaluator constructs the state from `robot0_gripper_qpos`, `robot0_eef_pos` and `robot0_eef_quat`. Preserve its image orientation, preprocessing and normalization conventions. [LIBERO evaluator](https://github.com/NVlabs/cosmos-policy/blob/main/cosmos_policy/experiments/robot/libero/run_libero_eval.py).

The published example enables `generate_future_state_and_value_in_parallel=True` in `get_action(...)`. Images are exposed as `result["future_image_predictions"]["future_image"]` and `result["future_image_predictions"]["future_wrist_image"]`. This gives us an existing image-output path to test. [Inference utilities](https://github.com/NVlabs/cosmos-policy/blob/main/cosmos_policy/experiments/robot/cosmos_utils.py).

## Downloads and the 45 GB correction

These are approximate **disk download sizes**, not peak GPU memory or the total installed environment.

| Component | Size | When needed |
|---|---:|---|
| `Cosmos-Policy-LIBERO-Predict2-2B.pt` | 3.91 GB | Policy and future-image prediction |
| `libero_t5_embeddings.pkl` | 42 MB | Supplied embeddings for standard LIBERO instructions |
| Complete policy repository | 3.95 GB | Above files plus configuration and dataset statistics |
| Image tokenizer/VAE and runtime dependencies | Additional | Required by the inference pipeline; not included in the policy repository total |
| Original full T5-11B PyTorch checkpoint | About 45.2 GB | Avoidable; includes encoder and decoder weights in FP32 |
| NVIDIA's T5 encoder-only release | About 9.73 GB | Candidate for encoding custom instructions without the full T5 download |

The policy weight file is 3,913,017,345 bytes; the supplied embedding cache is 41,957,938 bytes. [Policy files](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B/tree/main). The original T5 repository contains multiple weight formats; its total repository size is larger than the single PyTorch checkpoint. [Original T5 files](https://huggingface.co/google-t5/t5-11b/tree/main).

**The earlier suggestion that Cosmos necessarily requires a 45 GB text-encoder download was too strong.** That is a consequence of the stock checkpoint packaging, not a requirement to keep a 45 GB auxiliary model alongside the policy.

### Standard LIBERO task instructions

Use the supplied embedding cache. For matching instruction strings, inference does not need to load T5. The cache helper falls back to text encoding for an unknown string, so prompt spelling and cache coverage matter. [Cache implementation](https://github.com/NVlabs/cosmos-policy/blob/main/cosmos_policy/experiments/robot/cosmos_utils.py).

### Custom subtask instructions

NVIDIA publishes an encoder-only T5 package in the `text_encoder` directory of `Cosmos-Predict2-2B-Text2Image`: two safetensors shards totaling approximately 9.73 GB. This repository is gated. The Cosmos2 Diffusers implementation identifies its encoder as T5-11B; the conversion script saves it in BF16. [Encoder files](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image/tree/main/text_encoder), [Diffusers pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cosmos/pipeline_cosmos2_text2image.py), [conversion script](https://github.com/huggingface/diffusers/blob/main/scripts/convert_cosmos_to_diffusers.py).

Proposed integration: load only the encoder weights with the matching tokenizer, encode our finite list of subtask instructions, save the embeddings, and exit that process before running Cosmos Policy. Do not download the entire Text2Image pipeline just to obtain its encoder. This is a candidate loading path, **not a locally verified integration**.

Match the official encoding procedure: maximum 512 tokens, attention masking, last hidden states, and zeroed padding positions. The stock loader uses `T5EncoderModel` and `T5TokenizerFast` from `google-t5/t5-11b`; its default loader does not explicitly request BF16. Compare newly computed embeddings for a known instruction with the supplied cache before using custom prompts. [Official text encoding code](https://github.com/NVlabs/cosmos-policy/blob/main/cosmos_policy/_src/predict2/inference/get_t5_emb.py).

### Can a smaller text encoder replace T5?

An unrelated encoder is not a drop-in replacement just because its output dimensions match. The policy learned to interpret a particular embedding space. Using CLIP, Qwen or a smaller T5 would require a trained mapping or policy adaptation and validation; it is outside the first probe.

A community release offers the same old-T5 family in encoder-only FP16 (9.73 GB) and scaled FP8 (4.9 GB) formats. These require loader adaptation; the FP8 version also needs numerical and image-quality checks. Cosmos uses the older T5 variant, which differs from the T5 v1.1 commonly used with FLUX. [Community files](https://huggingface.co/comfyanonymous/cosmos_1.0_text_encoder_and_VAE_ComfyUI/tree/main/text_encoders), [ComfyUI Cosmos Predict2 documentation](https://comfyanonymous.github.io/ComfyUI_examples/cosmos_predict2/).

## Fit with our π0.7 experiment

Use Cosmos as a **frozen future-image provider**, with SmolVLA remaining the policy under comparison. This tests whether robot-trained visual predictions help our policy. It does not reproduce π0.7's BAGEL training recipe or all of Section V by itself.

Key limits for interpreting results:

- Cosmos predicts a future observation around a 16-action horizon. This is not necessarily the semantic end of the requested subtask. Do not equate it automatically with the paper's segment-end goals or assume a 40-step refresh interval is appropriate.
- The prediction is associated with Cosmos's own action model. SmolVLA may take a different trajectory; useful-looking goals are not evidence that SmolVLA can reach them.
- The checkpoint has already trained on LIBERO. Report this extra source of learned knowledge when comparing against SmolVLA. A gain would support the usefulness of the supplied goals, not isolate the benefit of prompt structure alone or establish unseen-task generalization.
- Custom subtask phrases need a prompt-sensitivity check. Correctly computing their embeddings does not establish that the policy understands these instructions as intended.
- Quality, speed, mistake labels, control-mode handling and prompt dropout still need to be implemented and evaluated in our policy experiment. Cosmos does not supply the missing dataset annotations.

Our downloaded `libero_10_subtask` contains two 256 × 256 cameras and an 8-D state. Cosmos expects the 9-D state described above. Verify the stored state semantics before converting orientation or reordering fields; padding 8 values to 9 is not a valid adapter. Simulator observations provide the explicit fields needed for an initial test. See the local [dataset audit](dataset.md).

## Budget assessment

Frozen inference is a reasonable candidate for the available 24 GB GPU. Upstream reports approximately 6.8 GB for basic policy inference, but that is not a measurement of our future-image configuration, Blackwell software stack, or concurrent use with SmolVLA. Record actual peak memory and latency before committing to an online evaluation. [Official repository](https://github.com/NVlabs/cosmos-policy).

Cache text embeddings ahead of time so T5 need not occupy GPU memory during image generation. Full Cosmos training is outside the current local budget: the published LIBERO training setup uses 64 H100 80 GB GPUs for about 48 hours. This does not establish the requirements of an untested lightweight adaptation method. [LIBERO training instructions](https://github.com/NVlabs/cosmos-policy/blob/main/LIBERO.md).

## Proposed first check

1. Install the official LIBERO inference environment and fetch the policy checkpoint, statistics, cached embeddings and required image tokenizer/VAE.
2. Use a simulator observation with both cameras and the exact 9-D state. Start with a standard task instruction present in the cache.
3. Enable future-state generation and save current and predicted external/wrist images, settings, seed, runtime and peak GPU memory.
4. Check physical plausibility, object identity, unchanged scene content and movement toward the instructed objective. Test several observations and instructions before deciding whether to integrate it.
5. If those results are useful, validate the encoder-only route and test our subtask prompts. Compare the known-task embedding against the official cache first.
6. Only then add a Cosmos-to-SmolVLA adapter and run matched policy comparisons, including the trained policy with subgoals omitted. Evaluate action outcomes as well as image quality.

**Status:** source and release-file review only. No Cosmos checkpoint download, local inference benchmark, encoder substitution test or SmolVLA integration was performed as part of this investigation. The next decision depends on the image-generation probe, not on the reported LIBERO policy score alone.
