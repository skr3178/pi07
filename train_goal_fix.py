"""todo.md tasks 1-2: goal-pathway capability test and fixes.

Same observations, batch size, seed and update count as goal_training_1000 (lr 1e-5 by default), but:
  - every training example has a goal, the real frame 8-16 steps ahead (capability setting);
  - optional fixes: a learned goal marker added to goal-image tokens (zero-initialised, so the start
    equals the base checkpoint) and unfreezing the last N VLM text layers (applied to A and B alike).
Scored with sampled actions on the 100 held-out observations of goal_training_1000, same noise as
evaluate_goal_pilot.py.
"""
import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from lerobot.policies.pretrained import PreTrainedPolicy

from goal_policy import GoalFlowMatching, configure_goals
from train_goal_pilot import (BASE, GOALS, ROOT, GoalIndex, GoalSamples, GoalSmolVLAPolicy, LeRobotDataset,
                              default_collate, flow_inputs, load_smolvla_config, make_pre_post_processors, np,
                              write_json)


class MarkedGoalFlowMatching(GoalFlowMatching):
    def __init__(self, config):
        super().__init__(config)
        hidden = self.vlm_with_expert.vlm.config.text_config.hidden_size
        self.goal_marker = torch.nn.Parameter(torch.zeros(hidden))

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state=None):
        embs, pads, blocks = super().embed_prefix(images, img_masks, lang_tokens, lang_masks, state)
        # Parent layout: [cur1, cur2, goal1, goal2] image tokens (equal counts), language, state.
        n_img = (embs.shape[1] - lang_tokens.shape[1] - 1) // 4
        is_goal = torch.zeros(embs.shape[1], 1, device=embs.device, dtype=embs.dtype)
        is_goal[2 * n_img:4 * n_img] = 1
        return embs + is_goal * self.goal_marker.to(embs.dtype) * math.sqrt(embs.shape[-1]), pads, blocks


class MarkedGoalPolicy(GoalSmolVLAPolicy):
    def __init__(self, config):
        PreTrainedPolicy.__init__(self, config)
        self.config = configure_goals(config)
        self.model = MarkedGoalFlowMatching(self.config)
        self.reset()


def load(marker, unfreeze_last, layout="codex", base=BASE):
    cfg, _ = load_smolvla_config(base)
    if layout == "alt":  # current images + text one block as trained; goals isolated after text
        from attention_check import AltGoalPolicy
        policy = AltGoalPolicy.from_pretrained(base, config=cfg, strict=True)
    elif marker:
        policy = MarkedGoalPolicy.from_pretrained(base, config=cfg, strict=False)
    else:
        policy = GoalSmolVLAPolicy.from_pretrained(base, config=cfg, strict=True)
    if unfreeze_last:
        layers = policy.model.vlm_with_expert.vlm.model.text_model.layers
        for layer in list(layers)[-unfreeze_last:]:
            for p in layer.parameters():
                p.requires_grad_(True)
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(base))
    return policy, pre, post


# Held-out modes: (goal source, change). Sources are the original validation goals (0-40 frames or segment
# end), goals 8-16 frames ahead, and the current subtask's last frame.
MODES = {"orig_real": ("orig", None), "near_real": ("near", None), "end_real": ("end", None),
         "near_wrong": ("near", "wrong"), "end_wrong": ("end", "wrong"),
         "copied_current": ("orig", "copy"), "dropped": ("orig", "drop")}
CAMS = ("observation.images.image", "observation.images.image2")


def evaluate(policy, pre, heldout, modes, partner):
    policy.eval(); results = {}
    for mode in modes:
        source, change = MODES[mode]; samples = heldout[source]
        sums = {8: 0.0, 50: 0.0}; counts = {8: 0, 50: 0}; per = {8: [], 50: []}
        for start in range(0, len(samples), 8):
            raw = default_collate(samples[start:start + 8])
            for cam, goal in zip(CAMS, GOALS):
                raw[f"{goal}_padding_mask"][:] = change != "drop"
                if change == "copy":
                    raw[goal] = raw[cam].clone()
                if change == "wrong":  # same goal type, from an observation of a different task
                    raw[goal] = torch.stack([samples[partner[i]][goal] for i in range(start, start + len(raw["task"]))])
            batch = pre(raw)
            noise, _ = flow_inputs(900000 + start, len(raw["task"]))
            with torch.inference_mode():
                pred = policy.predict_action_chunk(batch, noise=noise.cuda())
            err = (pred - batch["action"]).float().square().cpu(); valid = ~batch["action_is_pad"].cpu()
            for b in range(len(raw["task"])):
                for h in (8, 50):
                    v = err[b, :h][valid[b, :h]]; sums[h] += float(v.sum()); counts[h] += v.numel()
                    per[h].append(float(v.mean()))
        results[mode] = {"mse_8": sums[8] / counts[8], "mse_50": sums[50] / counts[50],
                         "per_obs_mse_8": per[8], "per_obs_mse_50": per[50]}
        print(f"  {mode}: mse_8 {results[mode]['mse_8']:.4f}  mse_50 {results[mode]['mse_50']:.4f}", flush=True)
    return results


def main(args):
    out = (ROOT / "artifacts/goal_fix" / args.variant); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    index = GoalIndex(ROOT / "data/libero_10_subtask")
    split = json.loads((ROOT / "artifacts/goal_training_1000/episode_split.json").read_text())
    n = args.steps * args.batch_size
    train = index.schedule(split["train_episodes"], n, args.seed, goal_probability=1.0)  # same `current` draws
    rng = np.random.default_rng(args.seed + 7)
    if args.goal == "near":
        train["future"] = np.minimum(train["current"] + rng.integers(8, 17, n), index.end[train["current"]])
        train["segment_endpoint"][:] = False
    else:
        train["future"] = index.segment_end[train["current"]]
        train["segment_endpoint"][:] = True
    ref = dict(np.load(ROOT / "artifacts/goal_training_1000/train_schedule.npz"))
    if n == len(ref["current"]):  # longer runs draw a different (still A/B-matched) schedule
        assert np.array_equal(ref["current"], train["current"]), "observation schedule must match goal_training_1000"
    validation = dict(np.load(ROOT / "artifacts/goal_training_1000/validation_schedule.npz"))
    if args.heldout_limit:  # smoke tests only
        validation = {k: v[:args.heldout_limit] for k, v in validation.items()}
    dataset = LeRobotDataset("lerobot/libero_10_subtask", root=index.root, video_backend="torchcodec",
                             delta_timestamps={"action": [i / index.info["fps"] for i in range(50)]})
    assert validation["goal_present"].all()
    cur = validation["current"]; val_rng = np.random.default_rng(args.seed + 11)
    goal_sets = {"orig": validation,
                 "near": validation | {"future": np.minimum(cur + val_rng.integers(8, 17, len(cur)), index.end[cur])},
                 "end": validation | {"future": index.segment_end[cur]}}
    heldout = {}
    for name, sched in goal_sets.items():
        s = GoalSamples(dataset, sched, "B"); heldout[name] = [s[i] for i in range(len(s))]
    tasks = index.tasks[cur]
    partner = [next((i + k) % len(cur) for k in range(10, 10 + len(cur)) if tasks[(i + k) % len(cur)] != tasks[i])
               for i in range(len(cur))]
    report = {"heldout": {"current": cur.tolist(), "episode": index.episodes[cur].tolist(), "task": tasks.tolist(),
                          "near_future": goal_sets["near"]["future"].tolist(), "end_future": goal_sets["end"]["future"].tolist(),
                          "wrong_task_partner": partner,
                          "old_shift10_same_task_pairs": int(sum(tasks[i] == tasks[(i + 10) % len(cur)] for i in range(len(cur))))},
              "variant": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()} | {"output": str(out)},
              "goal_rate": 1.0, "goal_offsets": {"near": "8-16 frames", "end": "current subtask's last frame"}[args.goal],
              "arms": {}}
    for arm in args.arms.split(","):
        torch.manual_seed(args.seed)
        policy, pre, post = load(args.marker, args.unfreeze_last, args.layout, args.base)
        marker_params = [p for n_, p in policy.named_parameters() if n_.endswith("goal_marker")]
        other = [p for n_, p in policy.named_parameters() if p.requires_grad and not n_.endswith("goal_marker")]
        groups = [{"params": other, "lr": args.lr}] + ([{"params": marker_params, "lr": args.marker_lr}] if marker_params else [])
        optimizer = torch.optim.AdamW(groups, betas=(.9, .95), eps=1e-8, weight_decay=1e-10)
        base_lrs = [g["lr"] for g in optimizer.param_groups]
        loader = DataLoader(GoalSamples(dataset, train, arm), batch_size=args.batch_size, shuffle=False, num_workers=0)
        torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter(); losses = []; curve = []
        curve_modes = ["dropped"] if arm == "A" else ["near_real", "end_real", "copied_current"]
        for step, raw in enumerate(loader, 1):
            batch = pre(raw)
            noise, times = flow_inputs(args.seed + 100000 + step, len(raw["task"]))
            for g, lr in zip(optimizer.param_groups, base_lrs):
                g["lr"] = lr * min(step / 20, 1.)
            policy.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = policy(batch, noise=noise.cuda(), time=times.cuda())
            loss.backward()
            if step == 1:  # unfrozen VLM layers and the marker must actually receive gradients
                vlm = [p for n_, p in policy.named_parameters() if p.requires_grad and ".vlm." in n_]
                report.setdefault("gradient_check", {})[arm] = {
                    "vlm_params_trainable": len(vlm),
                    "vlm_params_with_nonzero_grad": sum(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in vlm),
                    "marker_grad_nonzero": bool(marker_params and marker_params[0].grad is not None
                                                and marker_params[0].grad.abs().sum() > 0)}
                print(f"{arm} gradient check: {report['gradient_check'][arm]}", flush=True)
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 10., error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); losses.append(float(loss.detach()))
            if step % 250 == 0:
                print(f"{arm} step {step}: mean loss {np.mean(losses[-250:]):.4f}", flush=True)
            if args.eval_every and step % args.eval_every == 0 and step < args.steps:
                print(f"{arm} held-out at step {step}:", flush=True)
                curve.append({"step": step, "mean_loss_last250": float(np.mean(losses[-250:])),
                              "offline_action_mse": evaluate(policy, pre, heldout, curve_modes, partner)})
                write_json(out / f"curve_{arm}.json", curve)
        row = {"trainable_parameters": sum(p.numel() for g in groups for p in g["params"]),
               "base": str(args.base), "loss_per_250": [float(np.mean(losses[i:i + 250])) for i in range(0, len(losses), 250)],
               "curve": curve,
               "train_seconds": time.perf_counter() - t0, "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
               "final_mean_loss_last250": float(np.mean(losses[-250:]))}
        if marker_params:
            row["goal_marker_norm"] = float(marker_params[0].norm())
        print(f"{arm} trained: {json.dumps(row)}", flush=True)
        modes = ["dropped"] if arm == "A" else list(MODES)
        row["offline_action_mse"] = evaluate(policy, pre, heldout, modes, partner)
        report["arms"][arm] = row
        write_json(out / "report.json", report)
        if args.save_dir:  # after the report, so a save failure cannot lose results
            d = args.save_dir / args.variant / arm; d.mkdir(parents=True, exist_ok=False)
            policy.save_pretrained(d); pre.save_pretrained(d); post.save_pretrained(d)
            torch.save(optimizer.state_dict(), d / "optimizer.pt"); write_json(d / "run_args.json", report["variant"])
            print(f"{arm} saved to {d}", flush=True)
        del policy, pre, post, optimizer, groups, other, marker_params; gc.collect(); torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--arms", default="B")
    p.add_argument("--marker", action="store_true")
    p.add_argument("--layout", choices=["codex", "alt"], default="codex")
    p.add_argument("--unfreeze-last", type=int, default=0)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--marker-lr", type=float, default=1e-3)
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--goal", choices=["near", "end"], default="near")
    p.add_argument("--save-dir", type=Path)
    p.add_argument("--heldout-limit", type=int, default=0)
    main(p.parse_args())
