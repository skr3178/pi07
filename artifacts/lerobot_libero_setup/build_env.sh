#!/usr/bin/env bash
set -euo pipefail
SETUP_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SETUP_DIR/../.." && pwd)
PI07_ENV=${PI07_ENV:-/media/skr/storage/envs/pi07-lerobot}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/media/skr/storage/.uv_cache}
export UV_HTTP_TIMEOUT=600
export UV_CONCURRENT_DOWNLOADS=4
export UV_LINK_MODE=copy

if [[ ! -d "$PROJECT_DIR/vendor/lerobot/.git" ]]; then
  git clone --no-hardlinks /home/skr/Downloads/lerobot "$PROJECT_DIR/vendor/lerobot"
  git -C "$PROJECT_DIR/vendor/lerobot" checkout --detach 76a425c600b82e6bcf91650cbed347899eb1e07f
fi
if [[ ! -d "$PROJECT_DIR/vendor/lerobot-libero/.git" ]]; then
  git init "$PROJECT_DIR/vendor/lerobot-libero"
  git -C "$PROJECT_DIR/vendor/lerobot-libero" remote add origin https://github.com/huggingface/lerobot-libero.git
  git -C "$PROJECT_DIR/vendor/lerobot-libero" fetch --depth 1 origin 14fd6f022ff6647defc24e73563b95aa36670f43
  git -C "$PROJECT_DIR/vendor/lerobot-libero" checkout --detach FETCH_HEAD
fi
# Preserve local changes; apply the recorded runtime fix only when absent.
if git -C "$PROJECT_DIR/vendor/lerobot" apply --check "$SETUP_DIR/runtime_changes.patch" 2>/dev/null; then
  git -C "$PROJECT_DIR/vendor/lerobot" apply "$SETUP_DIR/runtime_changes.patch"
elif ! git -C "$PROJECT_DIR/vendor/lerobot" apply --reverse --check "$SETUP_DIR/runtime_changes.patch" 2>/dev/null; then
  echo "Runtime patch neither applies nor is already applied; inspect the checkout." >&2
  exit 1
fi
if [[ ! -x "$PI07_ENV/bin/python" ]]; then
  uv venv --python 3.11 "$PI07_ENV"
fi
uv pip install --python "$PI07_ENV/bin/python" \
  torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PI07_ENV/bin/python" \
  -c "$SETUP_DIR/constraints.txt" \
  -e "$PROJECT_DIR/vendor/lerobot[smolvla]" \
  -e "$PROJECT_DIR/vendor/lerobot-libero" gym==0.26.2
uv pip check --python "$PI07_ENV/bin/python"
uv pip freeze --python "$PI07_ENV/bin/python" > "$SETUP_DIR/env_freeze.txt"
python3 - "$PROJECT_DIR" "$PI07_ENV" "$SETUP_DIR" <<'PY'
import json, subprocess, sys
from pathlib import Path
project, env, out = map(Path, sys.argv[1:])
record = {'env': str(env), 'sources': {}}
for name in ('lerobot', 'lerobot-libero'):
    root = project / 'vendor' / name
    record['sources'][name] = {
        'path': str(root),
        'commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
    }
(out / 'source_commits.json').write_text(json.dumps(record, indent=2) + '\n')
PY
