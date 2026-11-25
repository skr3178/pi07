# Reserved final evaluation starts

[manifest.json](manifest.json) fixes 100 matched episodes: initial-state IDs
2–11 for every LIBERO-10 task. IDs 0 and 1 remain development-only. **These
final episodes have not run.** Pin the selected checkpoints, source hashes and
generation settings in a separate execution manifest before running them.

Dataset and simulator task IDs were matched by exact full task language:

| Simulator ID | Dataset ID |
| ---: | ---: |
| 0 | 5 |
| 1 | 6 |
| 2 | 0 |
| 3 | 1 |
| 4 | 7 |
| 5 | 9 |
| 6 | 8 |
| 7 | 4 |
| 8 | 3 |
| 9 | 2 |

The full task names and per-episode random seeds are in the manifest. The
available state arrays were checked to contain all requested IDs. Separation
from our development starts does not establish separation from either
released model's pretraining data.
