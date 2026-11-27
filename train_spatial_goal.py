"""Spatial subgoal test (plan_2026-11-25.md, Phase 1 step 1).

Same recipe as artifacts/goal_fix/l90v2_A (smolvla_libero_90, isolated goal layout, lr 1e-4, 2,000 updates,
batch 8, seed 20261029, identical observation and noise schedule), plus ONE extra prefix token: the robot
state (8 numbers: gripper position, rotation, finger opening) at the goal frame, normalized like the current
state. Goal images stay masked. If this beats A (0.421) clearly, goals are usable and the image pathway is
the bottleneck; if not, the policy can't exploit any goal at this budget.

The goal token sits in the goal block: it sees current images, text and itself; current images and text
can't see it; the state token and action expert see it. Its projection starts as a copy of state_proj.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

from attention_check import AltGoalFlowMatching, AltGoalPolicy
from lerobot.policies.pretrained import PreTrainedPolicy
from goal_policy import configure_goals
from train_goal_pilot import (ROOT, GoalIndex, GoalSamples, LeRobotDataset, default_collate, flow_inputs,
                              load_smolvla_config, make_pre_post_processors, np, write_json)

BASE = Path("/media/skr/storage/models/smolvla_libero_90_pi07")
GOAL_KEYS = ("goal_state", "goal_state_mask")


class SpatialGoalFlowMatching(AltGoalFlowMatching):
    def __init__(self, config):
        super().__init__(config)
        self.goal_state_proj = torch.nn.Linear(config.max_state_dim, self.state_proj.out_features)
        self.goal = None  # (B, max_state_dim) normalized goal state, (B,) bool present

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state=None):
        embs, pads, blocks = super().embed_prefix(images, img_masks, lang_tokens, lang_masks, state)
        goal, present = self.goal
        tok = self.goal_state_proj(goal.to(embs.device, self.goal_state_proj.weight.dtype))[:, None].to(embs.dtype)
        b = embs.shape[0]
        # Insert before the final (state) token, continuing the goal block (block flag 0).
        embs = torch.cat([embs[:, :-1], tok, embs[:, -1:]], 1)
        pads = torch.cat([pads[:, :-1], present.to(pads.device)[:, None], pads[:, -1:]], 1)
        blocks = torch.cat([blocks[:, :-1], torch.zeros(b, 1, dtype=blocks.dtype, device=blocks.device),
                            blocks[:, -1:]], 1)
        return embs, pads, blocks


class SpatialGoalPolicy(AltGoalPolicy):
    def __init__(self, config):
        PreTrainedPolicy.__init__(self, config)
        self.config = configure_goals(config)
        self.model = SpatialGoalFlowMatching(self.config)
        self.reset()

    def _set_goal(self, batch):
        self.model.goal = (batch["goal_state"].cuda(), batch["goal_state_mask"].cuda())

    def forward(self, batch, noise=None, time=None):
        self._set_goal(batch)
        return super().forward(batch, noise=noise, time=time)

    def predict_action_chunk(self, batch, noise=None):
        self._set_goal(batch)
        return super().predict_action_chunk(batch, noise=noise)


class StateTable:
    """Normalized, zero-padded robot state for any dataset row, read straight from parquet (no video decode)."""

    def __init__(self, root, max_state_dim):
        t = pq.read_table(root / "data", columns=["index", "observation.state"]).to_pandas().sort_values("index")
        assert np.array_equal(t["index"].to_numpy(), np.arange(len(t)))
        stats = load_file(BASE / "policy_preprocessor_step_5_normalizer_processor.safetensors")
        raw = torch.tensor(np.stack(t["observation.state"].to_numpy()), dtype=torch.float32)
        norm = (raw - stats["observation.state.mean"]) / (stats["observation.state.std"] + 1e-8)
        self.state = torch.nn.functional.pad(norm, (0, max_state_dim - norm.shape[1]))

    def __getitem__(self, rows):
        return self.state[rows]


class SpatialSamples(Dataset):
    """Arm-A samples (goal images zeroed and masked) plus the goal-frame state."""

    def __init__(self, dataset, schedule, states):
        self.base, self.schedule, self.states = GoalSamples(dataset, schedule, "A"), schedule, states

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        item["goal_state"] = self.states[int(self.schedule["future"][i])]
        item["goal_state_mask"] = torch.tensor(True)
        return item


def run_pre(pre, raw):
    goal = {k: raw.pop(k) for k in GOAL_KEYS}
    batch = pre(raw); batch.update(goal)
    return batch


def evaluate(policy, pre, heldout, goals, modes):
    """heldout: arm-A samples. goals[name] = (B, max_state_dim) goal states per held-out observation."""
    policy.eval(); results = {}
    for mode in modes:
        sums = {8: 0.0, 50: 0.0}; counts = {8: 0, 50: 0}; per = {8: [], 50: []}
        for start in range(0, len(heldout), 8):
            raw = default_collate(heldout[start:start + 8]); n = len(raw["task"])
            raw["goal_state"] = goals[mode if mode != "dropped" else "current"][start:start + n]
            raw["goal_state_mask"] = torch.full((n,), mode != "dropped")
            batch = run_pre(pre, raw)
            noise, _ = flow_inputs(900000 + start, n)
            with torch.inference_mode():
                pred = policy.predict_action_chunk(batch, noise=noise.cuda())
            err = (pred - batch["action"]).float().square().cpu(); valid = ~batch["action_is_pad"].cpu()
            for b in range(n):
                for h in (8, 50):
                    v = err[b, :h][valid[b, :h]]; sums[h] += float(v.sum()); counts[h] += v.numel()
                    per[h].append(float(v.mean()))
        results[mode] = {"mse_8": sums[8] / counts[8], "mse_50": sums[50] / counts[50],
                         "per_obs_mse_8": per[8], "per_obs_mse_50": per[50]}
        print(f"  {mode}: mse_8 {results[mode]['mse_8']:.4f}  mse_50 {results[mode]['mse_50']:.4f}", flush=True)
    return results


def main(args):
    out = ROOT / "artifacts/spatial_goal" / args.variant; out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    index = GoalIndex(ROOT / "data/libero_10_subtask")
    split = json.loads((ROOT / "artifacts/goal_training_1000/episode_split.json").read_text())
    n = args.steps * args.batch_size
    # Identical draws to train_goal_fix.py, so the observations match the saved A and B-near/B-end arms.
    train = index.schedule(split["train_episodes"], n, args.seed, goal_probability=1.0)
    rng = np.random.default_rng(args.seed + 7)
    if args.goal == "near":
        train["future"] = np.minimum(train["current"] + rng.integers(8, 17, n), index.end[train["current"]])
    else:
        train["future"] = index.segment_end[train["current"]]
    validation = dict(np.load(ROOT / "artifacts/goal_training_1000/validation_schedule.npz"))
    if args.heldout_limit:
        validation = {k: v[:args.heldout_limit] for k, v in validation.items()}
    dataset = LeRobotDataset("lerobot/libero_10_subtask", root=index.root, video_backend="torchcodec",
                             delta_timestamps={"action": [i / index.info["fps"] for i in range(50)]})
    cfg, _ = load_smolvla_config(BASE)
    states = StateTable(index.root, cfg.max_state_dim)

    cur = validation["current"]; val_rng = np.random.default_rng(args.seed + 11)  # same as train_goal_fix
    near = np.minimum(cur + val_rng.integers(8, 17, len(cur)), index.end[cur]); end = index.segment_end[cur]
    tasks = index.tasks[cur]
    partner = [next((i + k) % len(cur) for k in range(10, 10 + len(cur)) if tasks[(i + k) % len(cur)] != tasks[i])
               for i in range(len(cur))]
    goals = {"near_real": states[near], "end_real": states[end], "current": states[cur],
             "copied_current": states[cur], "near_wrong": states[near[partner]], "end_wrong": states[end[partner]]}
    heldout = GoalSamples(dataset, validation, "A"); heldout = [heldout[i] for i in range(len(heldout))]

    torch.manual_seed(args.seed)
    policy = SpatialGoalPolicy.from_pretrained(BASE, config=cfg, strict=False)
    policy.model.goal_state_proj.load_state_dict(policy.model.state_proj.state_dict())
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(BASE))
    params = [p for p in policy.parameters() if p.requires_grad]
    goal_params = sum(p.numel() for p in policy.model.goal_state_proj.parameters())
    assert all(p.requires_grad for p in policy.model.goal_state_proj.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(.9, .95), eps=1e-8, weight_decay=1e-10)
    own = f"{args.goal}_real"
    report = {"variant": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "base": str(BASE), "goal": {"near": "state 8-16 frames ahead", "end": "state at current subtask's last frame"}[args.goal],
              "goal_rate": 1.0, "trainable_parameters": sum(p.numel() for p in params), "goal_proj_parameters": goal_params,
              "heldout": {"current": cur.tolist(), "episode": index.episodes[cur].tolist(), "task": tasks.tolist(),
                          "near_future": near.tolist(), "end_future": end.tolist(), "wrong_task_partner": partner},
              "reference_A_mse_8": 0.4205}

    loader = DataLoader(SpatialSamples(dataset, train, states), batch_size=args.batch_size, shuffle=False, num_workers=0)
    torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter(); losses = []; curve = []
    for step, raw in enumerate(loader, 1):
        batch = run_pre(pre, raw)
        noise, times = flow_inputs(args.seed + 100000 + step, len(raw["task"]))
        for g in optimizer.param_groups:
            g["lr"] = args.lr * min(step / 20, 1.)
        policy.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy(batch, noise=noise.cuda(), time=times.cuda())
        loss.backward()
        if step == 1:
            g = policy.model.goal_state_proj.weight.grad
            report["goal_proj_grad_nonzero"] = bool(g is not None and g.abs().sum() > 0)
            print("goal projection receives gradient:", report["goal_proj_grad_nonzero"], flush=True)
        torch.nn.utils.clip_grad_norm_(params, 10., error_if_nonfinite=True)
        optimizer.step(); optimizer.zero_grad(set_to_none=True); losses.append(float(loss.detach()))
        if step % 250 == 0:
            print(f"step {step}: mean loss {np.mean(losses[-250:]):.4f}", flush=True)
        if args.eval_every and step % args.eval_every == 0 and step < args.steps:
            print(f"held-out at step {step}:", flush=True)
            curve.append({"step": step, "mean_loss_last250": float(np.mean(losses[-250:])),
                          "offline_action_mse": evaluate(policy, pre, heldout, goals, [own, "copied_current", "dropped"])})
            write_json(out / "curve.json", curve)
    report.update({"train_seconds": time.perf_counter() - t0, "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                   "loss_per_250": [float(np.mean(losses[i:i + 250])) for i in range(0, len(losses), 250)],
                   "final_mean_loss_last250": float(np.mean(losses[-250:])), "curve": curve})
    print("final held-out:", flush=True)
    report["offline_action_mse"] = evaluate(policy, pre, heldout, goals,
                                            ["near_real", "end_real", "copied_current", "near_wrong", "end_wrong", "dropped"])
    write_json(out / "report.json", report)
    if args.save_dir:
        d = args.save_dir / args.variant; d.mkdir(parents=True, exist_ok=False)
        policy.save_pretrained(d); pre.save_pretrained(d); post.save_pretrained(d)
        torch.save(optimizer.state_dict(), d / "optimizer.pt"); write_json(d / "run_args.json", report["variant"])
        print(f"saved to {d}", flush=True)
    del policy, optimizer; gc.collect(); torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--goal", choices=["near", "end"], required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-dir", type=Path)
    p.add_argument("--heldout-limit", type=int, default=0)
    main(p.parse_args())
