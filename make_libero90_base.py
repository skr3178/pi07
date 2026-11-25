"""Adapter folder for Kesvill/smolvla_libero_90 with smolvla_libero's camera names.

goal_policy.configure_goals expects observation.images.camera1/camera2. This writes a folder whose config
and preprocessor rename image/image2 to camera1/camera2 exactly as smolvla_libero does, drops the inactive
compile fields our loader rejects, and links the original weights and normalizer files (images use
IDENTITY normalization, so their stats are unused).
"""
import json
from pathlib import Path

SRC = Path("/media/skr/storage/models/smolvla_libero_90")
DST = Path("/media/skr/storage/models/smolvla_libero_90_pi07")
RENAME = {"observation.images.image": "observation.images.camera1",
          "observation.images.image2": "observation.images.camera2"}


def renamed(features):
    return {RENAME.get(k, k): v for k, v in features.items()}


def main():
    DST.mkdir(exist_ok=False)
    config = json.loads((SRC / "config.json").read_text())
    assert config.pop("compile_model") is False; config.pop("compile_mode")
    config["input_features"] = renamed(config["input_features"])
    (DST / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    pre = json.loads((SRC / "policy_preprocessor.json").read_text())
    for step in pre["steps"]:
        if step["registry_name"] == "rename_observations_processor":
            assert step["config"]["rename_map"] == {}; step["config"]["rename_map"] = RENAME
        if step["registry_name"] == "normalizer_processor":
            step["config"]["features"] = renamed(step["config"]["features"])
    (DST / "policy_preprocessor.json").write_text(json.dumps(pre, indent=2) + "\n")
    for name in ("model.safetensors", "policy_postprocessor.json",
                 "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
                 "policy_preprocessor_step_5_normalizer_processor.safetensors"):
        (DST / name).symlink_to(SRC / name)
    print("wrote", DST)


if __name__ == "__main__":
    main()
