# Generated training-goal pilot

20 ImageWAM predictions, two observations per LIBERO-10 task, drawn only from training-schedule observations with visible goals. The generator received current main/wrist images, current state and subtask text, with full-task fallback for unavailable phase text. One call predicts 16 transitions, not the segment endpoint.

All four review sheets compare current, generated +16, real near +16 and real segment endpoint. Real future frames were decoded after generation and were never generator inputs.

Assistant visual review: **14 accepted as plausible near continuations, 5 unresolved, 1 rejected**. This is a training-cache pilot, not an independent quality benchmark. Accepted images are not certified endpoints, counterfactual instruction-following examples or simulator-achievable states. Several images mostly continue arm motion and convey little new object-state information.

Use `reviewed_manifest.json` for the reviewed subset; the original `manifest.json` remains unchanged. Each decision has a per-image rationale. Rejected entries remain excluded even under the explicit unreviewed-data override.

Twenty cached observations cover only a small fraction of the 8,000-example schedule. Any mixture experiment must report the actual generated count; this cache cannot support a claimed 50% generated-goal training condition.
