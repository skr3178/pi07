"""Small, explicit compatibility helpers for the pinned LIBERO experiment."""

from dataclasses import fields
import json
from pathlib import Path

import draccus
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


def load_smolvla_config(checkpoint, *, device="cuda", n_action_steps=8):
    """Load the newer checkpoint config without editing it or hiding active features.

    Our older LeRobot has no PEFT/RTC options. Only their inactive values are
    compatible. All other unknown fields are errors rather than silently ignored.
    """
    source = json.loads((Path(checkpoint) / "config.json").read_text())
    if source.get("type") != "smolvla":
        raise ValueError("Expected a SmolVLA checkpoint")
    inactive = {"use_peft": False, "rtc_config": None}
    supported = {f.name for f in fields(SmolVLAConfig)}
    ignored = {"type": source.pop("type")}
    for key, expected in inactive.items():
        if key in source and key not in supported:
            if source[key] is not expected:
                raise ValueError(f"Pinned runtime cannot implement active {key}={source[key]!r}")
            ignored[key] = source.pop(key)
    unknown = set(source) - supported
    if unknown:
        raise ValueError(f"Unsupported checkpoint fields: {sorted(unknown)}")
    config = draccus.decode(SmolVLAConfig, source)
    config.device = device
    config.load_vlm_weights = False  # Complete backbone weights are in the policy checkpoint.
    config.n_action_steps = n_action_steps
    config.push_to_hub = False
    if not 1 <= config.n_action_steps <= config.chunk_size:
        raise ValueError("Execution length must be within the action chunk")
    return config, {"adapter": "inactive_fields_only", "handled_fields": ignored,
                    "overrides": {"device": device, "load_vlm_weights": False,
                                  "n_action_steps": n_action_steps, "push_to_hub": False}}
