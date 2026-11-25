# LIBERO-Goal action convention correction

The downloaded FastWAM dataset uses gripper targets `0 = close, 1 = open`.
Our SmolVLA/LIBERO interface uses `+1 = close, -1 = open`. The Goal adapter
now applies `command = 1 - 2 * target` before normalization. All six arm
commands and eight state values are preserved. This was corrected before
any LIBERO-Goal policy training; existing LIBERO-10 results are unaffected.

Evidence: the local ImageWAM evaluator applies this same transformation in
`ImageWAM/experiments/libero/eval_libero_single.py`, around lines 578–581.
The [official FastWAM integration documentation](https://huggingface.co/docs/lerobot/main/fastwam#libero-action-toggle)
also describes the gripper convention adjustment at execution.

The full local Goal data has raw gripper values exactly {0,1}. Across
within-episode transitions with finger-width change above 0.0001, raw 0
has mean width change -0.002776 (7,868 transitions), and raw 1 has +0.001669
(8,821 transitions), consistent with closing/opening respectively.

[report.json](report.json) checks 30 decoded observations across all ten
tasks, all corresponding 50-step action chunks and 520 padded positions.
State and arm targets match raw data exactly; transformed gripper targets,
camera shapes and padding match exactly. This supersedes the earlier
schema-only Goal check, which incorrectly accepted unchanged actions.
It does not certify a full simulator replay or policy success.
