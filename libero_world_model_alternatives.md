# LIBERO-trained world model alternatives

Research snapshot: 2026-10-29. Companion to [model.md](model.md) and the [Cosmos Policy report](cosmos_policy_report.md).

The [LIBERO-10 boundary test](artifacts/imagewam_boundary_probe/README.md) did not establish reliable instruction-driven redirection with ImageWAM. The [revised plan](reconstruction.md) retains it as a predictor of usual LIBERO-10 continuations and tests familiar-instruction control separately on LIBERO-Goal. These alternatives remain comparison or adaptation candidates; none has been validated locally.

## Proposed alternatives

| Model | Released LIBERO weights | What it generates | Fit for our experiment |
| --- | --- | --- | --- |
| **mimic-video** | Cosmos 2B fine-tuned separately on **Goal, Object and Spatial**; each video checkpoint is **3.91 GB** | Current image + instruction → future video | **Closest new candidate for subgoal images.** Released video checkpoints cover those three suites; no LIBERO-10/Long version was found in this search. [Weights](https://huggingface.co/jonpai/mimic-video/tree/main/video_backbone), [code](https://github.com/mimic-video/mimic-video) |
| **A2World / A2World-sim** | Released LIBERO-adapted checkpoint, built on Cosmos 2B | Current camera images + **20 robot actions** → future multiview video | Strong simulator candidate, but needs actions supplied first. [Model](https://huggingface.co/Fleurrr/A2World-World-Model), [code](https://github.com/LogosRoboticsGroup/A2World) |
| **WoVR / RLinf-Wan** | Suite-specific weights, including **LIBERO-Long** | Current image + task text + robot actions → future RGB frames | Useful for evaluating imagined action sequences. Published interface produces one camera view. [Documentation](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/wan.html), [Long weights](https://huggingface.co/RLinf/RLinf-Wan-LIBERO-Long/tree/main) |
| **RynnVLA-002**, successor to WorldVLA | Explicit world-model checkpoints, including **LIBERO-10** | Current images + robot actions → next main and wrist images | Matches our dataset more directly, but the LIBERO-10 world-model weights alone are approximately **14.1 GB**. [Code](https://github.com/alibaba-damo-academy/RynnVLA-002), [weights](https://huggingface.co/Alibaba-DAMO-Academy/RynnVLA-002/tree/main/World_model_512/libero_10) |

## How these fit the experiment

Our intended subgoal interface is **current observations + subtask instruction → goal images**, followed by a separate SmolVLA policy that selects robot actions. Among these alternatives, mimic-video is the closest interface match, initially on one of its supported LIBERO suites.

A2World, WoVR and RynnVLA-002 instead predict the consequences of supplied robot actions. Using them to choose subgoals would require a policy or planner to propose those actions first. A2World is the first candidate to investigate if we need an action-conditioned learned simulator.

## Verification status

- This search checked published code, interfaces and checkpoint availability; these alternatives have not been tested locally for this experiment.
- **24 GB inference compatibility is unverified.** Checkpoint file size is not runtime GPU memory: text encoders, image/video tokenizers and inference buffers add requirements.
- None has passed our instruction-control or LIBERO-10 subtask-boundary tests. Released LIBERO weights alone do not establish that a model understands our subtask wording or improves the downstream policy.
- Availability and file sizes are a dated snapshot; the source links above identify the releases examined.
