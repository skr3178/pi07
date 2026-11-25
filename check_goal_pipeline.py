"""Measured integration gates before any matched training run."""
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/goal_pipeline_checks"
OUT.mkdir(exist_ok=True)
os.environ.setdefault("HF_HOME", "/media/skr/storage/hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "artifacts/lerobot_libero_setup/datasets_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch.utils.data import default_collate
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from libero_runtime import load_smolvla_config
from goal_policy import GoalSmolVLAPolicy, GOALS
from goal_data import GoalIndex, GoalSamples, save_schedule

REPORT = {"status": "running"}


def save():
    (OUT / "report.json").write_text(json.dumps(REPORT, indent=2)+"\n")


def copy_batch(batch):
    return {k: v.clone() if torch.is_tensor(v) else v for k,v in batch.items()}


def main():
    torch.set_num_threads(4); torch.manual_seed(20261029)
    index = GoalIndex(ROOT / "data/libero_10_subtask")
    split = index.split()
    (OUT / "episode_split.json").write_text(json.dumps(split,indent=2)+"\n")
    schedule = index.schedule(split["train_episodes"], 10000, seed=42)
    schedule_hash = save_schedule(OUT / "audit_schedule.npz", schedule)
    present = schedule["goal_present"]
    present_count = int(present.sum())
    halfwidth = 3*np.sqrt(10000*.25*.75)
    assert abs(present_count-2500) <= halfwidth
    violations = int(np.sum(index.episodes[schedule["current"]] != index.episodes[schedule["future"]]))
    assert violations == 0
    assert not set(split["train_episodes"]) & set(split["validation_episodes"])
    assert np.all(np.isin(index.episodes[schedule["future"]], split["train_episodes"]))
    REPORT["data"] = {"pairs_checked": 10000, "cross_episode_pairs": violations,
        "train_episodes": len(split["train_episodes"]), "validation_episodes": len(split["validation_episodes"]),
        "goal_present_count": present_count, "goal_present_rate": present_count/10000,
        "segment_endpoint_fraction_given_goal": float(schedule["segment_endpoint"][present].mean()),
        "schedule_sha256": schedule_hash, "repaired_one_frame_artifact_episodes": len(index.repaired_artifact_episodes),
        "physical_initial_state_grouping_verified": False,
        "future_horizon": "0–40 retained dataset frames or current segment end; no physical-time equivalence claimed"}
    dataset = LeRobotDataset("lerobot/libero_10_subtask", root=index.root, video_backend="torchcodec",
                             delta_timestamps={"action": [i/10 for i in range(50)]})
    samples = GoalSamples(dataset,schedule,"B")
    # Decode a goal-present sample per task and compare each camera to its source.
    picked = []
    for task in range(10):
        idx = int(np.flatnonzero(present & (index.tasks[schedule["current"]]==task))[0])
        item = samples[idx]; future = dataset[int(schedule["future"][idx])]
        for goal,camera in zip(GOALS,("observation.images.image","observation.images.image2")):
            torch.testing.assert_close(item[goal], future[camera], atol=0, rtol=0)
        picked.append(idx)
    REPORT["data"]["goal_camera_pairs_exactly_matched"] = len(picked)
    raw = default_collate([samples[i] for i in picked[:2]])
    # Exercise mixed present/dropped batches, including masked slots retained for one sample.
    for goal in GOALS:
        raw[f"{goal}_padding_mask"] = torch.tensor([False,True])
    cfg, adapter = load_smolvla_config("/media/skr/storage/models/smolvla_libero")
    policy = GoalSmolVLAPolicy.from_pretrained("/media/skr/storage/models/smolvla_libero",config=cfg,strict=True)
    pre, _ = make_pre_post_processors(policy.config,pretrained_path="/media/skr/storage/models/smolvla_libero")
    batch = pre(raw)
    assert "action_is_pad" in batch
    params = [p for p in policy.parameters() if p.requires_grad]
    REPORT["model"] = {"parameters": sum(p.numel() for p in policy.parameters()),
                       "trainable_parameters": sum(p.numel() for p in params),
                       "backbone_trainable_parameters": sum(p.numel() for p in policy.model.vlm_with_expert.vlm.parameters() if p.requires_grad),
                       "adapter": adapter}
    assert REPORT["model"]["backbone_trainable_parameters"] == 0
    noise = torch.randn(2,50,32,device="cuda")
    policy.eval()
    with torch.inference_mode():
        first = policy.predict_action_chunk(copy_batch(batch),noise=noise.clone())
        changed = copy_batch(batch)
        for goal in GOALS:
            changed[goal][0] = torch.rand_like(changed[goal][0])
        second = policy.predict_action_chunk(changed,noise=noise.clone())
        hidden_change = float((first[0]-second[0]).abs().max())
        assert hidden_change <= 1e-6, hidden_change
        no_goals = copy_batch(batch)
        for goal in GOALS:
            no_goals[f"{goal}_padding_mask"][:] = False
        third = policy.predict_action_chunk(no_goals,noise=noise.clone())
        batch_invariance = float((first[0]-third[0]).abs().max())
        assert batch_invariance <= 1e-6, batch_invariance
        enabled_goal_change = float((first[1]-third[1]).abs().mean())
        images, masks = policy.prepare_images(batch)
        _, valid, boundaries = policy.model.embed_prefix(images,masks,batch['observation.language.tokens'],
            batch['observation.language.attention_mask'],policy.prepare_state(batch))
        full = make_att_2d_masks(valid,boundaries)
        obs_end = int(torch.where(boundaries[0])[0][0])
        assert not full[:,:obs_end,obs_end:].any()
    REPORT["masking"] = {"masked_pixels_action_max_delta": hidden_change,
        "other_batch_members_goal_mask_action_max_delta": batch_invariance,
        "valid_goals_action_mean_delta": enabled_goal_change,
        "current_observation_tokens": obs_end, "observation_to_goal_or_text_edges": 0}
    print("Data, strict loading, dropped-goal invariance and attention gates passed",flush=True)
    save()

    training = copy_batch(batch)
    training["action_is_pad"][:,-10:] = True
    t = torch.full((2,),.5,device="cuda")
    policy.train()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, details = policy(training,noise=noise.clone(),time=t)
        altered = copy_batch(training)
        altered['action'][:,-10:] += 1000
        loss_changed, _ = policy(altered,noise=noise.clone(),time=t)
    pad_delta = float((loss-loss_changed).abs().detach())
    assert pad_delta <= 1e-6, pad_delta
    del loss_changed
    loss.backward()
    grads = [p.grad for p in params if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    initial = float(loss.detach())
    grad_norm = float(torch.nn.utils.clip_grad_norm_(params,10))
    optimizer = torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-10)
    optimizer.zero_grad(set_to_none=True)
    # This 30-update fixed-noise/batch check tests learnability, not generalization.
    losses, seconds = [], []
    for step in range(30):
        begin=time.perf_counter()
        with torch.autocast("cuda",dtype=torch.bfloat16):
            loss,_=policy(training,noise=noise.clone(),time=t)
        assert torch.isfinite(loss)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(params,10)
        assert torch.isfinite(norm)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        losses.append(float(loss.detach())); seconds.append(time.perf_counter()-begin)
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        final=float(policy(training,noise=noise.clone(),time=t)[0])
    REPORT["training_gate"]={"steps":30,"batch_size":2,"padding_target_loss_delta":pad_delta,
        "initial_fixed_batch_loss":initial,"final_fixed_batch_loss":final,
        "relative_loss_reduction":(initial-final)/initial,"initial_gradient_norm":grad_norm,
        "gradient_tensors_checked":len(grads),"finite_gradient_tensors":len(grads),
        "step_seconds_p50":float(np.median(seconds)),"step_seconds_p95":float(np.percentile(seconds,95)),
        "peak_allocated_gb":torch.cuda.max_memory_allocated()/1e9,
        "peak_reserved_gb":torch.cuda.max_memory_reserved()/1e9,"losses":losses,
        "purpose":"fixed-batch/noise implementation check; weights discarded"}
    assert final < initial*.95, "Investigate fixed-batch learning before matched training"
    REPORT["status"]="passed"
    print(json.dumps(REPORT['training_gate']),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception:
        REPORT['status']='failed'; REPORT['error']=traceback.format_exc()
        raise
    finally:
        save()
