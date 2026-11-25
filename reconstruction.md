# Toy π0.7: LIBERO policy benefit and instruction control

**Active goal and resume point (2026-11-04): [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).**
The user has authorized implementing and evaluating the LIBERO recreation.
The newer plan reviews local assets and supersedes the next-step ordering below.
New code supports checkpoint-aware live evaluation, LIBERO-90 Section V training,
audited generated-goal mixtures and an explicit simulator-state subtask oracle.
See that plan for current measured gates and unfinished experiments.

Updated 2026-10-30 after implementing Section V prompt inputs and their training interface. Source: [π0.7 paper](pi_07.pdf), especially Section V, “Diversifying the Prompt.” Supporting notes: [dataset.md](dataset.md), [model.md](model.md), [equations.md](equations.md). The earlier PushT plan is [archived](artifacts/reconstruction_pusht_plan_2026-10-28.md); PushT is optional debugging, not the main experiment.

## Current progress and where to resume

Use the [quantitative scorecard](run_scorecard.md) at every stage. It records measurements, working criteria and when to continue, investigate or narrow a claim; it does not turn research outcomes into arbitrary pass/fail tests.

Saved checkpoint: **2026-10-30, Section V input code validated; existing baseline checkpoints ready, broader baseline scoring pending.** This is the main local plan to return to. Distinguish completed pilots from proposed final experiments.

| Step | Work | Status at this checkpoint |
| --- | --- | --- |
| 1 | Download LIBERO-10 subtask data and LIBERO-Goal; inspect observations, actions and labels | **Done.** Label exceptions are documented in [dataset.md](dataset.md). |
| 2 | Test ImageWAM image quality, language control and LIBERO-10 subtask boundaries | **Done.** Familiar LIBERO-Goal instruction control is supported; reliable LIBERO-10 subtask redirection is not. |
| 3 | Define separate LIBERO-10 visual-guidance and LIBERO-Goal instruction-control experiments | **Plan written.** Both are detailed below. |
| 4 | Set up LeRobot/LIBERO and verify dataset → policy → simulator | **Done for runtime preflight.** Strict weight load, dataset-to-action inference, both suites' camera rendering/action steps and 4/4 exact initial-state checks pass. Dataset decoding expanded to 30 samples across 10 tasks. |
| 5 | Evaluate the downloaded SmolVLA checkpoint as an initial reference | **Development pilot done:** LIBERO-10 1/6 successes, LIBERO-Goal 6/6; 0 runtime exceptions. This is separate from training baseline A. [Results](artifacts/smolvla_reference_pilot/README.md). |
| 6 | Implement goal-image inputs, episode-safe future-image pairs and goal dropout | **Done.** 10,000 pairs, 0 episode violations, 25.09% goals; dropped-goal and padding invariance pass. [Checks](artifacts/goal_pipeline_checks/README.md). |
| 7 | Train matched A and B on LIBERO-10 | **Done for development:** 200- and 1,000-update matched runs. At 1,000 updates, action MSE is A 0.2892; B real goals 0.3403; B dropped 0.2882. [Results](artifacts/goal_training_1000/README.md). |
| 8 | Compare A, B with generated goals and B with goals dropped on matched LIBERO-10 starts | **Development comparison done:** A 0/6, B dropped 0/6, B generated 0/6; 0 errors across 18 episodes. Final 100 starts per condition are [reserved](artifacts/libero10_evaluation_design/README.md), not run. [Results and next diagnostic](artifacts/goal_closed_loop_dev/README.md). |
| 9 | Train/evaluate the matched LIBERO-Goal instruction-control comparison | **Not started.** Only the offline ImageWAM probe is complete. |
| 10 | Add subtask-text and mixed-quality metadata ablations | **Input/training code implemented and checked; experiments pending.** Subtask, duration, metadata schema, control tag and dropout are supported. Quality/mistake annotations and mixed-quality data are still missing. [Implementation](artifacts/section_v_implementation/README.md). |

**Immediate resume point: establish broader baseline scores first.** The downloaded LIBERO-trained SmolVLA and the existing adapted A checkpoint are ready. Their six-start scores (1/6 and 0/6) are development measurements, not a reliable suite-wide benchmark.

1. Evaluate the unchanged reference and adapted A across all ten LIBERO-10 tasks on development starts 0/1, with identical settings/seeds. Keep reserved final starts 2–11 unused.
2. Run the queued goal-use capability diagnostic. The current B's real-goal action MSE (0.3403) is worse than dropped-goal MSE (0.2882); this does not isolate the cause or prove that B ignores goals. Preserve the old input format if reusing old A; train a matched task-only control if using the new Section V format.
3. Use the new [Section V implementation](artifacts/section_v_implementation/README.md) for task/subtask/goal/combined profiles. All new profiles share the same 128-token limit and `ee` control tag. Use identical text availability in controlled text × goal comparisons, then test the paper's conditional dropout recipe separately.
4. Add reviewed mixed-quality rollouts on matching tasks before claiming V-C quality/mistake conditioning. Current demos supply subtask IDs and recorded episode lengths, but no quality or mistake labels. Unknown fields stay absent.
5. Connect a declared human/controller subtask source for live text-conditioned tests; never replay demo boundary times. Freeze settings before the final benchmark. LIBERO-Goal instruction control remains a separate experiment.

The new task-only/full profiles have passed preparation on the same 8,000-example schedule. Full has 25.725% visible goals; quality/mistake coverage remains 0/8,000. A CPU forward/backward check and a separate one-update training/save/reload smoke passed. **No new production training or broad baseline evaluation ran during this implementation step.** The previous 200-/1,000-update checkpoints and all original source hashes remain unchanged. The one-update smoke checkpoint is not a baseline.

Environment: `/media/skr/storage/envs/pi07-lerobot`; original policy: `/media/skr/storage/models/smolvla_libero`. ImageWAM retains its separate environment. No job owned by this implementation step remains active. The mechanical gates pass; efficacy remains **investigate**. See [todo.md](todo.md) and the [scorecard](run_scorecard.md).

## Decision and evidence

Run two complementary experiments:

1. **LIBERO-10: do predicted future images help a policy complete multi-step tasks?** Use the existing ImageWAM checkpoint as a predictor of the usual continuation. It has not demonstrated reliable control by our LIBERO-10 subtask wording.
2. **LIBERO-Goal: can an instruction select the intended outcome in a shared scene, and can the policy reach that outcome?** ImageWAM already shows familiar-instruction control in an offline start-scene probe. Closed-loop benefit remains untested.

Neither experiment alone reproduces all of π0.7. In particular, familiar LIBERO-Goal instructions do not establish novel composition, overriding a learned task order, or cross-robot transfer. The metadata experiment remains a separate required component for Section V-C.

The [LIBERO-Goal probe](artifacts/imagewam_goal_language_probe/README.md) reproduced 63/90 reference-image task matches after one call and 59/60 after five calls, from three training-set starting scenes. These are image-classification scores, not simulator success. The [boundary probe](artifacts/imagewam_boundary_probe/README.md) found aligned prompts closer to the real continuation in 23/36 comparisons after one call and 18/36 after five. Its task-0 images support the failure to redirect the initial stove action; they also show that text can affect continuation after that action. Fixed-order scene cues are a plausible explanation, not an isolated causal finding.

**Adaptation decision:** do not fine-tune the world model yet. Fixed-order subtask data may improve vocabulary alignment or endpoints, but offers weak supervision for alternative choices from the same state. Reliable redirection requires instruction-dependent outcome examples and a matching evaluation. Collect alternative orders only for tasks whose prerequisites permit them; closing a container before inserting its object is not an equivalent valid order.

## Side by side: paper and toy

| Paper component | Our implementation or experiment | Claim supported / remaining gap |
| --- | --- | --- |
| V-A: subtask instructions | Implemented audited ID-to-text mapping, offline dataset adapter and explicit runtime request API | Initial image-only A/B keeps policy text identical. Live subtask selection needs a declared controller; demonstration frame labels are not available during rollout. |
| V-B: subgoal images | Frozen ImageWAM, current main/wrist images + measured 8-D state + instruction → predicted images; SmolVLA consumes the goal images | LIBERO-10 tests useful future guidance; LIBERO-Goal tests familiar-instruction control. Fixed-horizon predictions differ from the paper's segment-end generator. |
| V-C: quality, speed, mistakes | Implemented typed fields, missingness/provenance and annotation loader; episode-length inputs available; collect and review matched-task mixed-quality rollouts | Demonstrations alone do not test learning from mixed-quality data. Missing labels stay unknown. |
| V-D: control mode | Implemented fixed `ee` tag, never dropped; unsupported joint requests rejected | Preserve its semantics; do not claim mode switching. |
| V-E: prompt dropout | Implemented 25% goal presence, 30% subtask dropout conditional on goals, 15% whole metadata plus 5% per-field dropout | Evaluate the same trained B policy with and without goals. A zero-filled visible image is not automatically equivalent to dropping its tokens. |
| VI-C: real and generated training goals | Start with real future frames; then evaluate a declared mixture of cached generated goals if needed | Real-only training is a useful first condition, but omits the paper's generated-image augmentation and creates a train/test mismatch. |
| VII: runtime refresh | Generate from the latest actual observation on a declared simulator-step schedule | The paper refreshes on instruction change or after 4 seconds, asynchronously. This toy initially pauses simulation during inference and reports wall-clock cost separately. |
| High-level policy | Not needed for the initial task-text-only A/B; later use a validated subtask controller or explicitly labeled oracle condition | Do not imply that ImageWAM chooses the next subtask. |
| Model and training | Same SmolVLA checkpoint and flow-matching policy objective in both arms | A smaller reconstruction of prompt conditioning, not reproduction of the paper's entire architecture or pretraining. |

Hardware: one RTX PRO 4000 Blackwell, 24 GB VRAM. ImageWAM and SmolVLA have separate environments. Policy training has been measured at 4.714 GB allocated for batch size 8. Live integration measured ImageWAM at 9.505 GB and SmolVLA at 1.282 GB peak allocated in separate processes; their sum is not a device-wide peak-memory measurement. Worker startup peaks at 17.607 GB before the policy loads.

## Experiment 1: LIBERO-10 main A/B

**Primary question:** does adding generated future images improve closed-loop success when all other policy inputs and training conditions are matched?

| | A: baseline | B: visual-goal policy |
| --- | --- | --- |
| Starting weights | Local `lerobot/smolvla_libero` checkpoint | Identical checkpoint |
| Training trajectories | Same episode split of `lerobot/libero_10_subtask` | Identical trajectories and action targets |
| Policy observations | Main and wrist images, 8-D state, full task text | Same, plus goal-image tokens on 25% of examples |
| Test-time policy inputs | Current observations and full task text | Same, plus ImageWAM images generated from current observations, measured state and the same full task text |
| Subtask text / metadata | Absent in this first comparison | Absent in this first comparison |
| Training budget | Fixed updates, batches, optimizer, trainable modules and seed schedule | Same; additionally report extra image-processing and generator cost |
| Action settings | Retain checkpoint's 50-step output chunk; execute 8 before replanning initially | Same |

Using task text only in both arms isolates the visual-goal addition and avoids a live boundary detector in the first experiment. A task-only A versus a subtask-plus-image B would test the combined prompt package; it could not attribute a gain solely to images.

### Training goals and splits

- The saved split uses 450 training / 50 validation episodes, five validation episodes per task. The dataset lacks simulator initial-state IDs, so physical-start grouping is unverified; this limitation replaces the earlier proposed group split. Save split and label provenance. Both released models already saw LIBERO training data, so this offline holdout is not proof of unseen-model generalization.
- Use real future images first. For examples with goals, follow the paper's real-goal mixture: 25% segment-end frames, 75% bounded future frames. Keep camera identity, stay within the episode and log actual offsets. Do not call an endpoint successful without validation.
- The LeRobot dataset labels its video timeline 10 fps; ImageWAM's training data labels its timeline 20 fps and removes no-ops. Establish the mapping to simulator control steps before claiming that an offset equals 4 physical seconds. Until then, use explicit frame/transition counts and report their provenance.
- Repair or exclude task 3's one-frame id-15 artifact from segment labels without deleting its action/image trajectory. Handle unusual reversed/short segment sequences explicitly. Task 9 remains usable for overall task success but is excluded from comparisons requiring two subtasks.
- Keep real future frames strictly as training inputs or offline diagnostics. They are not available for live policy evaluation.
- If real-only training does not transfer to generated goals, test a second B trained with a fixed, recorded mixture of cached ImageWAM goals from training observations. Keep the trajectory/action supervision and training budget matched. This is a planned ablation, not an unreported change after seeing test results.

### Runtime goal generation

Begin with one ImageWAM call per refresh (16 retained training transitions ahead), refreshing every 16 simulator actions as a declared initial schedule. Use measured current state at each refresh. Compare longer chains only as a separate validation ablation before fixing the final evaluation settings.

Five-call chains are a candidate longer-horizon condition, not a default requirement. They take about 7 seconds on the measured GPU and hold state fixed while the predicted arm moves. Record this inconsistency and accumulated image error. Prediction horizon and goal-refresh interval are separate choices. No real future state may enter generation.

For the initial offline-paced simulator evaluation, pause simulation during inference and report success per simulator step as well as wall-clock time. Do not claim real-time operation or a 4-second wall-clock refresh rate from a 7-second chain.

### Evaluation and interpretation

- First run a small functional smoke test. Then evaluate A, B with generated goals, and the same B with goals dropped on the same initial-state IDs and random seeds.
- Initial benchmark budget: 10 episodes per task across all 10 tasks (100 per condition). Report per-task and aggregate success, paired outcome differences and uncertainty. Treat this as an initial estimate, not a high-powered result. Use additional training seeds before a strong causal claim.
- Primary metric: LIBERO simulator task success. Also report validated subtask completion where available, steps-to-success among successful episodes, timeout fraction and wall-clock cost. A short failure is not a speed improvement.
- Offline diagnostic on held-out episode groups: B action error with real future goals, generated goals and dropped goals. This helps locate failures but cannot alone decide whether the policy or generator caused a closed-loop failure.
- Check copied-current-image or mismatched-goal controls when needed to distinguish useful future information from merely adding image tokens.
- A B advantage supports visual guidance on LIBERO-10's usual continuations. It does not show that arbitrary subtask wording steers ImageWAM, and its images may still hallucinate or mislead the policy.

## Experiment 2: LIBERO-Goal instruction control

Use the already-downloaded LIBERO-Goal data at `/media/skr/storage/datasets/LIBERO-fastwam/lerobot_v30/libero_goal_no_noops_lerobot`. Its camera names and 512 px renders differ from the LeRobot LIBERO-10 data; establish the preprocessing mapping rather than mixing schemas implicitly.

1. Retain the existing offline same-image/different-instruction probe as evidence of familiar language control. Add unseen simulator initial states and several seeds. Score object/goal conditions with simulator state or reviewed semantic labels, not only global pixel distance.
2. Train or adapt matched A/B policies on the same LIBERO-Goal trajectory set. Keep their text, observations and training budgets identical apart from goal inputs, as in Experiment 1.
3. Compare requested-goal success under different valid instructions from matched physical starting scenes. Verify scene/state compatibility when transferring initial states across task definitions. Include the same B with generated goals and with goals dropped.
4. Treat paraphrases and novel combinations as separate held-out conditions. The current 59/60 result uses familiar training instructions and does not establish these capabilities.

The generator's image-level sensitivity and the policy's requested-goal success are separate outcomes. Both must be measured before claiming that subgoals improve instruction following.

## Later Section V experiments

**Subtask text:** use a 2×2 comparison (text absent/present × goal absent/present) if attributing gains to each addition. A validated controller must choose the next subtask online. Simulator predicates or human instructions can support a declared oracle/controller-assisted condition, but never replay demonstration boundary times into a new rollout as if they were observations.

**Metadata:** add mixed-quality rollouts on the same tasks, retain failures and recoveries, and annotate quality, speed and mistakes with confidence and missingness. Compare nested top-quality subsets with and without metadata, holding all other prompt fields and update budgets fixed. Apply the paper's metadata dropout (15% whole prompt, 5% per field) and investigate CFG separately. Label injected perturbations as perturbations until their effects are validated as mistakes.

**World-model adaptation:** only if the intended interface fails. Fixed-order segment pairs can improve endpoint prediction or wording, but a steering claim requires scenes with different instruction-dependent outcomes. LIBERO-Goal and feasible reordered LIBERO-10 rollouts are candidate sources. Keep adaptation and evaluation episodes separate and retain both positive and negative instruction controls.

## Section V implementation status and order (2026-10-30)

Status of each "Diversifying the Prompt" part (paper Sec. V-A to V-E) in this toy, and whether it can be built with the current data and hardware. Effort figures are rough estimates, not measurements.

| Part | What the paper does | Our status | Can we do it? |
| --- | --- | --- | --- |
| **V-A Subtask instructions** | "Task: …" plus "Subtask: …" text in the prompt | Label-to-text mapping audited ([dataset.md](dataset.md)); not in the policy prompt yet | **Yes, data ready.** Needs a prompt builder and SmolVLA's `tokenizer_max_length` raised from 48 (the toy prompt below is about 55 tokens). At test time, something must name the current subtask, e.g. a controller that switches when the simulator's subtask predicate is satisfied (a declared oracle/controller-assisted condition). About ½–1 day. |
| **V-B Subgoal images** | World-model goal images; goals in 25% of training examples | Built and integrated, **not helping**: real future goals raised B's held-out action error (0.340 vs 0.288 dropped) | **Fix first.** The paper reports that goal images make training *significantly faster* (action prediction becomes an inverse-dynamics problem). Our opposite result points to the goal pathway, not the idea. Run the attention check and capability test in [todo.md](todo.md) (about 20 min). |
| **V-C Episode metadata** | Speed (episode length, 500-step bins), quality 1–5, per-segment mistake flag; learned from mixed-quality data | Not started | **Partly now.** Speed comes free from episode lengths (use about 50-step bins; LIBERO episodes are roughly 100–500 steps). Quality and mistakes need varied data: LIBERO demonstrations are all successful. Collect simulator rollouts (successes, failures, slow runs, recoveries), score with a documented rubric, and mark mistake segments. Largest piece: about 2–4 days. |
| **V-D Control mode** | Joint vs end-effector actions, tagged in the prompt | One end-effector action convention | **Skip for the toy.** Joint mode needs joint-space action data and a simulator controller change; a constant tag tests nothing. Document the omission. |
| **V-E Full prompt + dropout** | Goals in 25% of examples; subtask dropped 30% when a goal is present; metadata dropped 15% whole plus 5% per field; no dropout for control mode; metadata CFG at test time (β 1.3 / 1.7 / 2.2) | Only the 25% goal rule exists | **Yes, cheap** once V-A exists (same prompt builder). Metadata CFG needs two policy passes per prediction. |

Prompt format:

```
Paper:  <cameras><goal images> Task: peel vegetables. Subtask: pick up the peeler.
        Speed: 8000. Quality: 5. Mistake: false. Control Mode: joint.<robot state>
Toy:    <cameras><goal images> Task: put both moka pots on the stove.
        Subtask: put the first moka pot on the stove. Speed: 250. Quality: 5.
        Mistake: false.<robot state>
```

Recommended order:

1. **Fix the goal pathway (V-B):** attention check, then capability test ([todo.md](todo.md)). The other parts build on B working.
2. **Prompt builder with subtask text and dropout (V-A + V-E):** train matched variants (task only / + subtask / + goal / + both) for the 2×2 comparison above.
3. **A trustworthy baseline:** all 10 LIBERO-10 tasks and a longer training budget, so comparisons aren't at zero success.
4. **Metadata (V-C):** speed first, then collect and label mixed-quality simulator rollouts for quality and mistakes.
5. **Control mode (V-D):** documented as omitted.

## Implementation stages

1. **Runtime preflight:** isolated LeRobot + LIBERO environment, pinned source revisions, CUDA smoke test, AV1 dataset decoding, main/wrist/state/action mapping, headless simulator reset/step and one SmolVLA action. Artifacts: [environment setup](artifacts/lerobot_libero_setup/README.md).
2. **Baseline evaluation:** unmodified pretrained policy on a few tasks, then the selected seed manifest. This establishes a reference, not the fine-tuned A baseline.
3. **Goal-conditioning implementation:** explicit goal-image fields/tokens and missing-goal masks, episode-safe sampling, paper-inspired attention boundaries verified against actual SmolVLA code. Do not assume adding another camera key implements the paper's attention scheme.
4. **Matched policy training and LIBERO-10 A/B:** train both arms with the declared settings, then evaluate generated-goal and dropped-goal conditions.
5. **LIBERO-Goal steering experiment:** reuse the policy interface with a separate matched data split and requested-goal evaluation.
6. **Subtask / metadata ablations:** add the missing Section V components without conflating them with the initial image-only comparison.

Do not start long training until dataset decoding, one policy forward pass, one simulator step and observation/action conventions pass. Store source revisions, model paths, dataset revisions, settings and timings with every run.

## References and retained artifacts

- [π0.7 paper](pi_07.pdf): Sections V, VI-C and VII define the prompt components, real/generated goal training and refresh behavior.
- [Dataset selection and label audit](dataset.md).
- [ImageWAM initial probe](artifacts/imagewam_subgoal_probe/README.md), [prompt controls](artifacts/imagewam_prompt_control_probe/README.md), [LIBERO-Goal probe](artifacts/imagewam_goal_language_probe/README.md), [boundary probe](artifacts/imagewam_boundary_probe/README.md).
- [Alternative LIBERO world models](libero_world_model_alternatives.md) and [Cosmos Policy report](cosmos_policy_report.md).
- [Earlier PushT reconstruction plan](artifacts/reconstruction_pusht_plan_2026-10-28.md); its unvalidated assumptions and old environment commands are historical, not current instructions.
