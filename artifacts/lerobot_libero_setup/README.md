# LeRobot + LIBERO setup handoff

Updated 2026-10-29. Main [plan](../../reconstruction.md), [quantitative scorecard](../../run_scorecard.md), and [pretrained-policy pilot](../smolvla_reference_pilot/README.md).

## Runtime status: functional checks complete

- Isolated environment: `/media/skr/storage/envs/pi07-lerobot`. Python 3.11.14, PyTorch 2.7.1 + CUDA 12.8, torchvision 0.22.1, torchcodec 0.5, transformers 4.56.1, LeRobot 0.3.4, LIBERO 0.1.4 and MuJoCo 3.3.2. Dependency checks pass; [package list](env_freeze.txt).
- [Smoke report](smoke_report.json): strict load of all checkpoint weights succeeded (450,046,176 parameters), dataset-to-policy action passed, and both LIBERO suites rendered both cameras and accepted a policy action.
- Four requested initial-state checks (two per suite): maximum absolute error **0.0** before settling.
- [Dataset input audit](input_audit.json): **30/30** decoded samples, across **10 tasks**, all finite; **0** cross-episode action queries and **0** padding-mask mismatches. This samples start/middle/end of one episode per task; it does not certify every frame or the future-goal sampler.
- [Reference pilot](../smolvla_reference_pilot/report.json): **12/12** full episodes ran without exceptions; **23,751/23,751** action components were finite. LIBERO-10 **1/6** successes; LIBERO-Goal **6/6**. This is a development pilot, not the final benchmark or trained baseline A.
- Measured simulator control frequency: **20 Hz**. Dataset video fps labels do not establish cross-dataset physical time alignment.

## Compatibility changes

The complete policy weights already exist at `/media/skr/storage/models/smolvla_libero`. Only additional VLM configuration/tokenizer assets were fetched; all backbone weights come from the policy checkpoint.

[libero_runtime.py](../../libero_runtime.py) adapts the newer checkpoint configuration to the pinned loader. It verifies `type="smolvla"`, accepts only inactive `use_peft=false` and `rtc_config=null`, and rejects any other unsupported fields. The original checkpoint is unchanged. Strict weight loading succeeded. [Adapter checks](config_adapter_checks.json) reject active PEFT, active RTC and an unknown option. Initial failure logs are retained under `history/`.

The isolated LeRobot checkout has a [reset-order patch](runtime_changes.patch): reset the environment first, then apply the requested initial state. The four exact-state checks above validate this fix. The user's existing LeRobot checkout and ImageWAM environment were not modified.

The unbatched wrapper rotates camera images into negative-stride NumPy views. The pilot makes them contiguous before tensor conversion, matching the copying normally done by vector environments. Camera renaming and normalization use the checkpoint's saved processors.

## Resume after goal-conditioning integration

The goal adapter, sampler and padding fix are implemented and validated in
[goal pipeline checks](../goal_pipeline_checks/README.md). A
[matched 200-update pilot](../goal_training_pilot/README.md) is complete.
Use the [main plan](../../reconstruction.md) and [scorecard](../../run_scorecard.md)
for the current longer training and simulator development results.

The local training wrapper consumes `action_is_pad`, excludes padded timesteps
and unused action dimensions, and retains fixed goal slots with explicit masks.
Both arms share these changes and the same attention layout. The unmodified
reference pilot remains a separate pre-adaptation reference.

## Commands and provenance

From `/home/skr/Downloads/pi_07`:

```bash
/media/skr/storage/envs/pi07-lerobot/bin/python -u \
  pi07_code/artifacts/lerobot_libero_setup/smoke_test.py \
  > pi07_code/artifacts/lerobot_libero_setup/smoke.log 2>&1

/media/skr/storage/envs/pi07-lerobot/bin/python \
  pi07_code/artifacts/lerobot_libero_setup/audit_training_inputs.py
```

These commands rerun functional checks, not training. The scripts overwrite current reports; archive them first when preserving a comparison. The pilot runner refuses to overwrite its existing manifest; choose a new output directory for a new pilot.

- [Source revisions](source_commits.json): LeRobot `76a425c600b82e6bcf91650cbed347899eb1e07f`; LIBERO `14fd6f022ff6647defc24e73563b95aa36670f43`.
- [Build script](build_env.sh), [constraints](constraints.txt), [build log](build.log), [package list](env_freeze.txt).
- [Pilot runner](run_reference_pilot.py) and [log](reference_pilot.log).
- Installation is complete. For current run activity and the resume point, consult the main plan rather than this historical setup report.
