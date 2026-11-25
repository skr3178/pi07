"""Task 0 in todo.md: does the goal attention layout change the unmodified checkpoint's actions?

Untrained base checkpoint, 100 held-out observations of goal_training_1000, goals masked:
  original      plain SmolVLA (images and text in one bidirectional block)
  codex_layout  GoalSmolVLAPolicy (current images cannot attend to text)
  alt_layout    [current images, text] one block as trained; goals in a later block that sees
                images + text but is not seen by them; state last. With goals masked this should
                reproduce `original`.
"""
import json
import math
from pathlib import Path

import torch
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from goal_policy import GoalFlowMatching, configure_goals
from train_goal_pilot import (BASE, GOALS, ROOT, GoalIndex, GoalSamples, GoalSmolVLAPolicy, LeRobotDataset,
                              default_collate, flow_inputs, load_smolvla_config, make_pre_post_processors, np,
                              write_json)


class AltGoalFlowMatching(GoalFlowMatching):
    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state=None):
        embs, pads, blocks = [], [], []

        def add(value, pad, first_block):
            embs.append(value); pads.append(pad)
            blocks.extend([first_block] + [0] * (value.shape[1] - 1))

        for img, mask in zip(images[:2], img_masks[:2]):  # current cameras
            value = self.vlm_with_expert.embed_image(img)
            add(value * math.sqrt(value.shape[-1]), mask[:, None].expand(value.shape[:2]), 0)
        lang = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        add(lang * math.sqrt(lang.shape[-1]), lang_masks, 0)
        for n, (img, mask) in enumerate(zip(images[2:], img_masks[2:])):  # goals: new block after text
            value = self.vlm_with_expert.embed_image(img)
            add(value * math.sqrt(value.shape[-1]), mask[:, None].expand(value.shape[:2]), int(n == 0))
        st = self.state_proj(state)[:, None, :]
        add(st, torch.ones(st.shape[:2], dtype=torch.bool, device=st.device), 1)
        return (torch.cat(embs, 1), torch.cat(pads, 1),
                torch.tensor(blocks, dtype=torch.bool, device=st.device)[None].expand(st.shape[0], -1))


class AltGoalPolicy(GoalSmolVLAPolicy):
    def __init__(self, config):
        PreTrainedPolicy.__init__(self, config)
        self.config = configure_goals(config)
        self.model = AltGoalFlowMatching(self.config)
        self.reset()


def evaluate(policy, pre, samples, schedule, index):
    sums = {8: 0.0, 50: 0.0}; counts = {8: 0, 50: 0}; per_task = {t: [] for t in range(10)}; preds = []
    for start in range(0, len(samples), 8):
        raw = default_collate(samples[start:start + 8])
        for goal in GOALS:
            raw[f"{goal}_padding_mask"][:] = False
        batch = pre(raw)
        noise, _ = flow_inputs(900000 + start, len(raw["task"]))  # same noise as evaluate_goal_pilot.py
        with torch.inference_mode():
            pred = policy.predict_action_chunk(batch, noise=noise.cuda())
        err = (pred - batch["action"]).float().square().cpu(); valid = ~batch["action_is_pad"].cpu()
        preds.append(pred.float().cpu())
        for b in range(len(raw["task"])):
            for h in (8, 50):
                v = err[b, :h][valid[b, :h]]; sums[h] += float(v.sum()); counts[h] += v.numel()
            per_task[int(index.tasks[int(schedule["current"][start + b])])].append(float(err[b, :8][valid[b, :8]].mean()))
    return ({"mse_8": sums[8] / counts[8], "mse_50": sums[50] / counts[50],
             "per_task_mse_8": {t: float(np.mean(v)) for t, v in per_task.items()}}, torch.cat(preds))


def main():
    out = ROOT / "artifacts/attention_check"; out.mkdir(exist_ok=False)
    index = GoalIndex(ROOT / "data/libero_10_subtask")
    schedule = dict(np.load(ROOT / "artifacts/goal_training_1000/validation_schedule.npz"))
    dataset = LeRobotDataset("lerobot/libero_10_subtask", root=index.root, video_backend="torchcodec",
                             delta_timestamps={"action": [i / index.info["fps"] for i in range(50)]})
    samples = GoalSamples(dataset, schedule, "B")
    samples = [samples[i] for i in range(len(samples))]
    report, preds = {"checkpoint": str(BASE), "observations": len(samples), "goals": "masked in all layouts"}, {}
    for name, cls in (("original", SmolVLAPolicy), ("codex_layout", GoalSmolVLAPolicy), ("alt_layout", AltGoalPolicy)):
        cfg, _ = load_smolvla_config(BASE)
        policy = cls.from_pretrained(BASE, config=cfg, strict=True).eval()
        pre, _ = make_pre_post_processors(cfg, pretrained_path=str(BASE))
        report[name], preds[name] = evaluate(policy, pre, samples, schedule, index)
        print(name, json.dumps({k: v for k, v in report[name].items() if k != "per_task_mse_8"}), flush=True)
        del policy; torch.cuda.empty_cache()
    for name in ("codex_layout", "alt_layout"):
        report[f"max_abs_action_diff_{name}_vs_original"] = float((preds[name] - preds["original"]).abs().max())
    write_json(out / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k.startswith("max_abs")}), flush=True)


if __name__ == "__main__":
    main()
