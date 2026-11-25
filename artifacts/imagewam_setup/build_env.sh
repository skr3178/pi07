#!/usr/bin/env bash
# Build the ImageWAM Python env unattended (survives the Claude session closing).
# Result: env_status.txt and env_freeze.txt next to this script.
set -u
cd /home/skr/Downloads/pi_07/ImageWAM
export UV_CACHE_DIR=/media/skr/storage/.uv_cache UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=4
E=/media/skr/storage/envs/imagewam-venv
L=/media/skr/storage/envs/imagewam_setup.log
OUT=/home/skr/Downloads/pi_07/pi07_code/artifacts/imagewam_setup
check() {
  "$E/bin/python" -c "import torch, imagewam" >/dev/null 2>&1
}

# Let any earlier install into this env finish first.
while pgrep -f "[u]v pip install --python $E" >/dev/null; do sleep 15; done

status="already working"
if ! check; then
  status="failed after 5 attempts"
  for attempt in 1 2 3 4 5; do
    if { rm -rf "$E" && uv venv --python 3.11 "$E" \
         && uv pip install --python "$E/bin/python" -e ".[shared]" \
         && uv pip install --python "$E/bin/python" "transformers==4.56.1"; } > "$L" 2>&1; then
      status="built on attempt $attempt"
      break
    fi
    sleep 60
  done
fi

{
  echo "env: $status at $(date '+%F %T')"
  "$E/bin/python" -c "import torch, torchvision, transformers, diffusers, huggingface_hub; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| sm_120 supported:', 'sm_120' in torch.cuda.get_arch_list(), '| torchvision', torchvision.__version__, '| transformers', transformers.__version__, '| diffusers', diffusers.__version__, '| huggingface_hub', huggingface_hub.__version__)" 2>&1 | tail -1
  "$E/bin/python" -c "import imagewam; print('imagewam importable from', imagewam.__file__)" 2>&1 | tail -1
  du -sh "$E"
  echo "full log: $L"
} > "$OUT/env_status.txt" 2>&1
uv pip freeze --python "$E/bin/python" > "$OUT/env_freeze.txt" 2>/dev/null
