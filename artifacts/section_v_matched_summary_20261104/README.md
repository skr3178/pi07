# Section V matched training measurements

Same starting weights, observation/future schedule, optimizer, 1,000-update budget and 128-token interface verified.

| Profile | Updates/status | Final first-eight MSE | Goals off MSE |
| --- | --- | --- | --- |
| task | 1000 / complete | 0.5151 | — |
| subtask | 1000 / complete | 0.5139 | — |
| goal | 1000 / complete | 0.5661 | 0.4934 |
| subtask_goal | 1000 / complete | 0.5864 | 0.5103 |
| full | 1000 / complete | 0.5868 | 0.5171 |

Normalized action error, lower is better. These values do not measure live task success.

The four task/text/goal/combined profiles form the controlled comparison. The full profile also changes conditional text dropout and adds duration, so its difference cannot be attributed to metadata alone. Quality/mistake training on the separate curated mixture is a different experiment.

[report.json](report.json) contains learning curves, image-content controls and paired task-bootstrap intervals where available.
