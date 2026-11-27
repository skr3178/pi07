#!/bin/bash
# After the toy training queue: train a saved text-only policy, record matched rollouts, build comparison videos.
set -e
cd /home/skr/Downloads/pi_07/pi07_code
PY=/media/skr/storage/envs/pi07-lerobot/bin/python
LOG=artifacts/toy/logs
while pgrep -f "toy/toy_section_v.py --name t[123]_" > /dev/null; do sleep 30; done
echo "queue done $(date)"
[ -f artifacts/toy/live_text/model.pt ] || $PY toy/toy_section_v.py --name live_text --text 1 --seed 0 --save > $LOG/live_text.log 2>&1
echo "text policy saved $(date)"
# Smoke: one episode must run end to end before the full set
$PY toy/rollout.py --name smoke --policy live_goal --goal demo_end --tasks 0 --starts 1 --max-steps 40 > $LOG/rollout_smoke.log 2>&1
rm -rf artifacts/toy/rollouts/smoke
echo "smoke ok $(date)"
$PY toy/rollout.py --name text_only --policy live_text --goal none > $LOG/rollout_text_only.log 2>&1 &
$PY toy/rollout.py --name goal_end --policy live_goal --goal demo_end > $LOG/rollout_goal_end.log 2>&1 &
$PY toy/rollout.py --name goal_track --policy live_goal --goal demo_track > $LOG/rollout_goal_track.log 2>&1 &
wait
$PY toy/rollout.py --name goal_wrong --policy live_goal --goal demo_end --goal-task next > $LOG/rollout_goal_wrong.log 2>&1 &
$PY toy/rollout.py --name goal_imagewam --policy live_goal --goal imagewam > $LOG/rollout_goal_imagewam.log 2>&1 &
wait
echo "rollouts done $(date)"
$PY toy/compare_videos.py --runs "text_only=Instruction only,goal_end=Goal: demo end,goal_track=Goal: demo 4 s ahead,goal_imagewam=Goal: ImageWAM,goal_wrong=Goal: wrong task"
echo "all done $(date)"
