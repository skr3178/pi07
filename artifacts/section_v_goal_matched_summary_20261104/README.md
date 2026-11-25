# Section V matched training measurements

Dataset: libero_goal. Same starting weights, observation/future schedule, optimizer, 1000-update budget and 128-token interface verified.

| Profile | Updates/status | Final first-eight MSE | Goals off MSE |
| --- | --- | --- | --- |
| task | 1000 / complete | 0.6230 | — |
| goal | 1000 / complete | 0.6300 | 0.6260 |

Normalized action error, lower is better. These values do not measure live task success.

Conditional subtask dropout settings match across these profiles.

[report.json](report.json) contains learning curves, image-content controls and paired task-bootstrap intervals where available.
