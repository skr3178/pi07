"""Toy Section V testbed: a ~10M-parameter flow-matching policy trained from scratch on LIBERO-Goal.

Purpose: test π0.7's prompt ideas where each prompt part is NECESSARY and nothing is frozen.
LIBERO-Goal has one kitchen and 10 instructions, so the camera alone can't say which task to do.
  --text     task instruction (a learned embedding per instruction stands in for language)
  --goal     subgoal images, both cameras, same encoders as the current views. Training goals follow the
             paper: 25% end-of-segment frame, 75% a frame up to 4 s ahead (Sec. V-B / VI)
  --quality  quality metadata 1-5 (Sec. V-C) with synthetic mixed-quality action labels (--mix mixed)
  --dropout paper   goal present 25%; text dropped 30% given a goal; metadata dropped 15% + 5% (Sec. V-E)
Scored by sampled-action MSE (normalized units) on 50 held-out episodes, same noise for every arm.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE = Path("/media/skr/storage/datasets/pi07_toy/libero_goal_96")
OUT = Path(__file__).resolve().parent.parent / "artifacts/toy"
H, ADIM, DEV = 16, 7, "cuda"  # 16 actions = 0.8 s at 20 fps
QUALITY = {5: (1.0, 0.0), 3: (0.75, 0.25), 1: (0.5, 0.5)}  # level -> (arm-motion scale, noise in action stds)


class Data:
    def __init__(self, mix):
        m = np.load(CACHE / "meta.npz")
        self.main = torch.from_numpy(np.load(CACHE / "main.npy")).to(DEV)
        self.wrist = torch.from_numpy(np.load(CACHE / "wrist.npy")).to(DEV)
        episode, task = m["episode"], m["task"]
        rng = np.random.default_rng(0)  # split and quality labels are fixed across model seeds
        held = []
        for k in np.unique(task):
            eps = np.unique(episode[task == k]); rng.shuffle(eps); held += list(eps[:5])
        self.heldout_episodes = sorted(int(e) for e in held)
        train_eps = np.array(sorted(set(np.unique(episode)) - set(held)))
        level = dict(zip(rng.permutation(train_eps), np.resize([5, 3, 1], len(train_eps))))
        is_train = np.isin(episode, train_eps)
        raw = m["action"].astype(np.float32)
        mean, std = raw[is_train].mean(0), raw[is_train].std(0) + 1e-6
        quality = np.full(len(raw), 5); labels = raw.copy()
        if mix != "clean":
            noise = np.zeros((len(raw), 6), dtype=np.float32); eps_ = rng.standard_normal((len(raw), 6)).astype(np.float32)
            for i in range(1, len(raw)):  # smooth AR(1) noise, unit variance
                noise[i] = 0.9 * noise[i - 1] + math.sqrt(1 - 0.81) * eps_[i]
            for e in train_eps:
                rows = episode == e; scale, sigma = QUALITY[level[e]]; quality[rows] = level[e]
                labels[rows, :6] = scale * raw[rows, :6] + sigma * std[:6] * noise[rows]
        keep = is_train & (quality == 5) if mix == "clean_third" else is_train
        st = m["state"].astype(np.float32); smean, sstd = st[is_train].mean(0), st[is_train].std(0) + 1e-6
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=DEV)
        self.state = t((st - smean) / sstd); self.state_mean, self.state_std = t(smean), t(sstd)
        self.clean, self.labels = t((raw - mean) / std), t((labels - mean) / std)
        self.act_mean, self.act_std = t(mean), t(std)
        self.task, self.quality, self.episode = t(task, torch.long), t(quality, torch.long), t(episode, torch.long)
        self.ep_start, self.ep_end = t(m["ep_start"], torch.long), t(m["ep_end"], torch.long)
        self.train_frames = t(np.flatnonzero(keep), torch.long)
        ev = np.flatnonzero(np.isin(episode, held))[::3]  # every 3rd held-out frame
        self.eval_frames = t(ev, torch.long)
        # wrong-task goal: last frame of a held-out episode of the next task
        by_task = {k: [e for e in self.heldout_episodes if task[episode == e][0] == k] for k in np.unique(task)}
        ends = {e: int(m["ep_end"][episode == e][0]) for e in self.heldout_episodes}
        nxt = lambda k: sorted(by_task)[(sorted(by_task).index(k) + 1) % len(by_task)]
        self.wrong_goal = t([ends[by_task[nxt(task[i])][int(episode[i]) % 5]] for i in ev], torch.long)

    def chunk(self, t, source):
        idx = t[:, None] + torch.arange(H, device=DEV); end = self.ep_end[t][:, None]
        return source[idx.clamp(max=end)], idx > end

    def train_batch(self, n, g):
        t = self.train_frames[torch.randint(len(self.train_frames), (n,), device=DEV, generator=g)]
        end = self.ep_end[t]
        fut = torch.minimum(t + torch.randint(0, 81, (n,), device=DEV, generator=g), end)  # up to 4 s ahead
        return t, torch.where(torch.rand(n, device=DEV, generator=g) < 0.25, end, fut)   # 25%: end of segment


def shift(x, pad=4):
    n, c, h, w = x.shape
    x = F.pad(x, (pad,) * 4, mode="replicate"); ar = torch.arange(h, device=x.device)
    iy, ix = torch.randint(0, 2 * pad + 1, (2, n), device=x.device)
    rows = (iy[:, None] + ar)[:, :, None].expand(n, h, w); cols = (ix[:, None] + ar)[:, None, :].expand(n, h, w)
    return x[torch.arange(n, device=x.device)[:, None, None], :, rows, cols].permute(0, 3, 1, 2)


class Encoder(nn.Module):
    def __init__(self, out=256):
        super().__init__()
        layers, c = [], 3
        for i, k in enumerate((32, 64, 128, 256)):  # 96 -> 48 -> 24 -> 12 -> 6
            layers += [nn.Conv2d(c, k, 5 if i == 0 else 3, 2, 2 if i == 0 else 1), nn.GroupNorm(8, k), nn.SiLU()]; c = k
        self.conv = nn.Sequential(*layers); self.head = nn.Linear(256 * 36, out)

    def forward(self, x):
        return self.head(self.conv(x).flatten(1))


def mlp(i, w, o, n=2):
    layers, c = [], i
    for _ in range(n):
        layers += [nn.Linear(c, w), nn.SiLU()]; c = w
    return nn.Sequential(*layers, nn.Linear(c, o))


class Policy(nn.Module):
    def __init__(self, a, d=256, c=512):
        super().__init__()
        self.a = a
        self.enc_main, self.enc_wrist, self.state = Encoder(d), Encoder(d), mlp(8, d, d, 1)
        parts = 3
        if a.text:
            self.text = nn.Embedding(10, d); self.null_text = nn.Parameter(torch.zeros(d)); parts += 1
        if a.quality:
            self.qual = nn.Embedding(5, d); self.null_qual = nn.Parameter(torch.zeros(d)); parts += 1
        if a.goal:
            self.goal = mlp(2 * d, d, d, 1); self.null_goal = nn.Parameter(torch.zeros(d)); parts += 1
        self.fuse = mlp(parts * d, 1024, c, 2)
        self.head = mlp(H * ADIM + 64 + c, 1024, H * ADIM, 3)

    def images(self, d, idx, train):
        out = []
        for store, enc in ((d.main, self.enc_main), (d.wrist, self.enc_wrist)):
            x = store[idx].float()
            out.append(enc((shift(x) if train else x) / 255 - 0.5))
        return out

    def context(self, d, t, fut, use, quality, train=False):
        parts = self.images(d, t, train) + [self.state(d.state[t])]
        if self.a.text:
            parts.append(torch.where(use["text"][:, None], self.text(d.task[t]), self.null_text))
        if self.a.quality:
            parts.append(torch.where(use["quality"][:, None], self.qual(quality - 1), self.null_qual))
        if self.a.goal:
            parts.append(torch.where(use["goal"][:, None], self.goal(torch.cat(self.images(d, fut, train), 1)), self.null_goal))
        return self.fuse(torch.cat(parts, 1))

    def velocity(self, x, time, ctx):
        f = torch.exp(-math.log(1000) * torch.arange(32, device=DEV) / 32); ang = 1000 * time[:, None] * f
        return self.head(torch.cat([x, ang.sin(), ang.cos(), ctx], 1))

    def loss(self, d, t, fut, use, g):
        action, pad = d.chunk(t, d.labels)
        n = len(t); noise = torch.randn(n, H * ADIM, device=DEV, generator=g)
        time = torch.rand(n, device=DEV, generator=g) ** (1 / 1.5) * 0.999 + 0.001  # Beta(1.5, 1), π0 convention
        x = time[:, None] * noise + (1 - time[:, None]) * action.flatten(1)
        err = (self.velocity(x, time, self.context(d, t, fut, use, d.quality[t], True)) - (noise - action.flatten(1))) ** 2
        valid = (~pad)[:, :, None].expand(-1, -1, ADIM).flatten(1)
        return (err * valid).sum() / valid.sum()

    @torch.no_grad()
    def sample(self, d, t, fut, use, quality, noise, beta=0.0, steps=10):
        ctx = self.context(d, t, fut, use, quality)
        if beta:  # classifier-free guidance on the metadata: v_uncond + beta * (v_cond - v_uncond)
            off = self.context(d, t, fut, {**use, "quality": torch.zeros_like(use["quality"])}, quality)
        x = noise.clone()
        for i in range(steps):
            time = torch.full((len(t),), 1 - i / steps, device=DEV); v = self.velocity(x, time, ctx)
            if beta:
                vu = self.velocity(x, time, off); v = vu + beta * (v - vu)
            x = x - v / steps
        return x.view(-1, H, ADIM)


def train_masks(a, n, g):
    r = lambda: torch.rand(n, device=DEV, generator=g)
    on = torch.ones(n, dtype=torch.bool, device=DEV)
    if a.dropout == "none":
        return {"text": on, "goal": on, "quality": on}
    goal = r() < 0.25 if a.goal else on
    text = ~(goal & (r() < 0.30)) if a.goal else on
    return {"text": text, "goal": goal, "quality": (r() >= 0.15) & (r() >= 0.05)}


def eval_modes(a):
    out = {}
    for text in ([True, False] if a.text and a.dropout == "paper" and a.goal else [bool(a.text)]):
        for goal in (["end", "near", "copy", "wrong", "off"] if a.goal else ["off"]):
            out[f"text{int(text)}_goal-{goal}"] = {"text": text, "goal": goal, "quality": 5, "beta": 0.0}
    if a.quality:
        base = {"text": bool(a.text), "goal": "end" if a.goal and a.dropout == "none" else "off"}
        for q in (3, 1, 0):
            out[f"quality-{q or 'off'}"] = {**base, "quality": q, "beta": 0.0}
        if a.dropout == "paper":
            for beta in (1.3, 1.7, 2.2):
                out[f"quality-5_cfg{beta}"] = {**base, "quality": 5, "beta": beta}
    return out


@torch.no_grad()
def evaluate(model, d, modes, noise):
    model.eval(); t_all = d.eval_frames; res = {}
    for name, m in modes.items():
        e8, e16, num, den = [], [], 0.0, 0.0
        for s in range(0, len(t_all), 512):
            t = t_all[s:s + 512]; n = len(t); end = d.ep_end[t]
            fut = {"end": end, "near": torch.minimum(t + 40, end), "copy": t, "off": t, "wrong": d.wrong_goal[s:s + 512]}[m["goal"]]
            on = torch.ones(n, dtype=torch.bool, device=DEV)
            use = {"text": on & m["text"], "goal": on & (m["goal"] != "off"), "quality": on & bool(m["quality"])}
            pred = model.sample(d, t, fut, use, torch.full((n,), m["quality"] or 5, device=DEV), noise[s:s + 512], m["beta"])
            target, pad = d.chunk(t, d.clean)
            err = ((pred - target) ** 2).mean(-1).masked_fill(pad, 0); valid = (~pad).float()
            e8.append(err[:, :8].sum(1) / valid[:, :8].sum(1)); e16.append(err.sum(1) / valid.sum(1))
            # gain: how much of the clean arm motion the prediction reproduces (1 = full size, 0.5 = half size)
            p = (pred * d.act_std + d.act_mean)[..., :6] * valid[..., None]; q = (target * d.act_std + d.act_mean)[..., :6] * valid[..., None]
            num += float((p * q).sum()); den += float((q * q).sum())
        e8, e16 = torch.cat(e8), torch.cat(e16)
        early = (t_all - d.ep_start[t_all]) < 40  # first 2 s: the scene looks the same for every task
        res[name] = {"mse_8": float(e8.mean()), "mse_16": float(e16.mean()), "mse_8_first2s": float(e8[early].mean()),
                     "gain": num / den, "per_obs_mse_8": e8.cpu().tolist()}
    model.train()
    return res


def main(a):
    out = OUT / a.name; out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(a.seed); torch.backends.cudnn.benchmark = True; torch.set_num_threads(2)
    d = Data(a.mix); model = Policy(a).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    lr = lambda s: min(s / 200, 1) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / a.steps)))
    g = torch.Generator(device=DEV).manual_seed(a.seed)
    noise = torch.randn(len(d.eval_frames), H * ADIM, device=DEV, generator=torch.Generator(device=DEV).manual_seed(12345))
    modes = eval_modes(a); main_mode = next(iter(modes))
    report = {"args": vars(a), "parameters": sum(p.numel() for p in model.parameters()), "train_frames": len(d.train_frames),
              "eval_observations": len(d.eval_frames), "eval_episode": d.episode[d.eval_frames].cpu().tolist(),
              "eval_task": d.task[d.eval_frames].cpu().tolist(), "curve": []}
    t0 = time.perf_counter(); losses = []
    for step in range(1, a.steps + 1):
        for group in opt.param_groups:
            group["lr"] = 3e-4 * lr(step)
        t, fut = d.train_batch(a.batch, g)
        loss = model.loss(d, t, fut, train_masks(a, a.batch, g), g)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 10.); opt.step()
        losses.append(float(loss))
        if step % a.eval_every == 0 and step < a.steps:
            r = evaluate(model, d, {main_mode: modes[main_mode]}, noise)[main_mode]
            report["curve"].append({"step": step, "loss": float(np.mean(losses[-a.eval_every:])), "mse_8": r["mse_8"]})
            print(f"step {step}: loss {report['curve'][-1]['loss']:.4f}  {main_mode} mse_8 {r['mse_8']:.4f}", flush=True)
    report["train_seconds"] = time.perf_counter() - t0; report["final_loss"] = float(np.mean(losses[-500:]))
    report["eval"] = evaluate(model, d, modes, noise)
    for k, v in report["eval"].items():
        print(f"  {k}: mse_8 {v['mse_8']:.4f}  first2s {v['mse_8_first2s']:.4f}  gain {v['gain']:.3f}", flush=True)
    (out / "report.json").write_text(json.dumps(report))
    if a.save:
        torch.save({"model": model.state_dict(), "args": vars(a), "heldout_episodes": d.heldout_episodes,
                    "act_mean": d.act_mean.cpu(), "act_std": d.act_std.cpu(),
                    "state_mean": d.state_mean.cpu(), "state_std": d.state_std.cpu()}, out / "model.pt")
    print(f"done in {report['train_seconds']:.0f}s, {report['parameters'] / 1e6:.1f}M parameters", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--text", type=int, default=0)
    p.add_argument("--goal", type=int, default=0)
    p.add_argument("--quality", type=int, default=0)
    p.add_argument("--dropout", choices=["none", "paper"], default="none")
    p.add_argument("--mix", choices=["clean", "clean_third", "mixed"], default="clean")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", action="store_true")
    main(p.parse_args())
