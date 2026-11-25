#!/usr/bin/env bash
# Wait for the detached ImageWAM checkpoint download, list model.pt's contents (no weights loaded,
# no pickled code run), and delete orphaned partial downloads only if model.pt is complete.
# Result: checkpoint_contents.txt next to this script.
set -u
M=/media/skr/storage/models/imagewam
D=$M/imagewam_release/libero/flux2_klein_4b
OUT=/home/skr/Downloads/pi_07/pi07_code/artifacts/imagewam_setup
EXPECTED=9035379577   # model.pt size on Hugging Face, bytes

until [ -f "$M/download_imagewam_ckpt.done" ]; do sleep 30; done

{
  echo "download: $(cat "$M/download_imagewam_ckpt.done")"
  size=$(stat -c %s "$D/model.pt" 2>/dev/null || echo 0)
  echo "model.pt: $size bytes (expected $EXPECTED)"
  if [ "$size" = "$EXPECTED" ]; then
    echo "complete: yes"
    n=$(ls "$D"/.cache/huggingface/download/*.incomplete 2>/dev/null | wc -l)
    rm -f "$D"/.cache/huggingface/download/*.incomplete
    echo "orphaned partial files deleted: $n"
    echo
    /usr/bin/python3 -I "$OUT/inspect_pt.py" "$D/model.pt"
  else
    echo "complete: NO - partial files kept; rerun the download:"
    echo "  hf download yuyangalin/ImageWAM-FLUX.2-4B-LIBERO --local-dir $D"
  fi
} > "$OUT/checkpoint_contents.txt" 2>&1
