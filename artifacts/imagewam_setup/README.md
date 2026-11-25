# ImageWAM FLUX.2 4B LIBERO: setup record

Set up 2026-10-28 for the world-model test. **Not tested yet:** the GPU was in use.

## Local paths

| Item | Path |
|---|---|
| Code | `~/Downloads/pi_07/ImageWAM` (commit in `source_commits.txt`). FLUX.2 source in `third_party/flux2` at the pinned commit. |
| Local config | `ImageWAM/.env.local` (copy: `env.local.copy`) |
| Local code change | `pyproject.toml` torch index switched from CUDA 11.8 to 12.8 for the Blackwell GPU (`runtime_changes.patch`) |
| Python env | `/media/skr/storage/envs/imagewam-venv` (linked as `ImageWAM/.venv`); Python 3.11, built with `uv pip install -e ".[shared]"` then `transformers==4.56.1` |
| Checkpoint | `/media/skr/storage/models/imagewam/imagewam_release/libero/flux2_klein_4b/` (`model.pt`, `config.yaml`, `dataset_stats.json`) |
| FLUX.2 Klein base 4B | `/media/skr/storage/models/imagewam/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors` |
| Autoencoder | `/media/skr/storage/models/imagewam/flux2/FLUX.2-dev/ae.safetensors` (gated repo; license accepted on `sangramrout`) |
| Qwen3-4B | Hugging Face cache at `/media/skr/storage/hf_home` |

`/media/skr/storage/models` and `/media/skr/storage/hf_home` are links to the Seagate drive (`/media/skr/SeagateHub1`). The env and the `uv` cache are on the storage drive itself.

## Status (checked 2026-10-29)

All three detached jobs finished:

- **Download:** `model.pt` complete (9,035,379,577 bytes, exit 0 at 19:50 on 2026-10-28). The orphaned 5.9 GB partial file was deleted.
- **Env:** built on the first attempt. torch 2.7.1+cu128, sm_120 supported, `imagewam` importable; 7.6 GB. Versions are in `env_freeze.txt`.
- **Checkpoint:** `model.pt` holds the full fine-tuned FLUX.2 Klein 4B transformer (3.876B params) plus a 642M action head and the proprio encoder (4.518B unique params). Details are in `checkpoint_contents.txt`.

## Jobs left running when the session closed

All three were detached so they'd survive the session closing:

| Job | Writes when done |
|---|---|
| Checkpoint download (restarted after a stall) | `/media/skr/storage/models/imagewam/download_imagewam_ckpt.done` (exit code + time); log `download_imagewam_ckpt.log` beside it |
| `build_env.sh`: env build, up to 5 attempts | `env_status.txt` (torch/CUDA/import checks) and `env_freeze.txt` here; full log `/media/skr/storage/envs/imagewam_setup.log` |
| `check_checkpoint.sh`: waits for the download, then lists `model.pt` | `checkpoint_contents.txt` here. If `model.pt` matches the Hugging Face size (9,035,379,577 bytes), it also deletes the orphaned partial download (~5.9 GB) left by the stalled first attempt. |

To check from a terminal: `cat pi07_code/artifacts/imagewam_setup/{env_status,checkpoint_contents}.txt`.

## Open questions for the test

- ~~Whether `model.pt` holds the full fine-tuned FLUX.2 model plus the action head.~~ Yes; see Status.
- What the image path conditions on and how far ahead it predicts. The config has `num_frames: 17` and a 224×448 input (two cameras side by side). Read the paper (arXiv 2606.19531) before testing.
- Estimated GPU need: ~11–13 GB without Qwen3 loaded. Encode the prompts first, then unload Qwen3.
