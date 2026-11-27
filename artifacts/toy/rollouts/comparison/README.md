# Toy policy in the LIBERO-Goal simulator: comparison videos

Same task and starting scene in every column. Top: live camera. Bottom: the goal image given to the policy.

| Task | Start | Instruction | Instruction only | Goal: demo end | Goal: demo 4 s ahead | Goal: ImageWAM | Goal: wrong task |
|---|---|---|---|---|---|---|---|
| 0 | 0 | [open the middle drawer of the cabinet](task0_start0.mp4) | ✓ | ✓ | ✓ | ✗ | ✗ |
| 0 | 1 | [open the middle drawer of the cabinet](task0_start1.mp4) | ✗ | ✓ | ✓ | ✓ | ✗ |
| 1 | 0 | [put the bowl on the stove](task1_start0.mp4) | ✓ | ✓ | ✓ | ✓ | ✗ |
| 1 | 1 | [put the bowl on the stove](task1_start1.mp4) | ✓ | ✓ | ✓ | ✓ | ✗ |
| 2 | 0 | [put the wine bottle on top of the cabinet](task2_start0.mp4) | ✓ | ✗ | ✗ | ✗ | ✗ |
| 2 | 1 | [put the wine bottle on top of the cabinet](task2_start1.mp4) | ✓ | ✓ | ✓ | ✓ | ✗ |
| 3 | 0 | [open the top drawer and put the bowl inside](task3_start0.mp4) | ✓ | ✗ | ✓ | ✗ | ✗ |
| 3 | 1 | [open the top drawer and put the bowl inside](task3_start1.mp4) | ✓ | ✗ | ✗ | ✗ | ✗ |
| 4 | 0 | [put the bowl on top of the cabinet](task4_start0.mp4) | ✓ | ✓ | ✓ | ✓ | ✗ |
| 4 | 1 | [put the bowl on top of the cabinet](task4_start1.mp4) | ✓ | ✓ | ✓ | ✓ | ✗ |
| 5 | 0 | [push the plate to the front of the stove](task5_start0.mp4) | ✗ | ✗ | ✓ | ✓ | ✗ |
| 5 | 1 | [push the plate to the front of the stove](task5_start1.mp4) | ✗ | ✓ | ✓ | ✗ | ✗ |
| 6 | 0 | [put the cream cheese in the bowl](task6_start0.mp4) | ✗ | ✗ | ✗ | ✗ | ✗ |
| 6 | 1 | [put the cream cheese in the bowl](task6_start1.mp4) | ✓ | ✗ | ✓ | ✓ | ✗ |
| 7 | 0 | [turn on the stove](task7_start0.mp4) | ✓ | ✓ | ✓ | ✓ | ✗ |
| 7 | 1 | [turn on the stove](task7_start1.mp4) | ✓ | ✓ | ✓ | ✓ | ✗ |
| 8 | 0 | [put the bowl on the plate](task8_start0.mp4) | ✓ | ✓ | ✓ | ✓ | ✗ |
| 8 | 1 | [put the bowl on the plate](task8_start1.mp4) | ✓ | ✓ | ✓ | ✓ | ✗ |
| 9 | 0 | [put the wine bottle on the rack](task9_start0.mp4) | ✓ | ✓ | ✓ | ✓ | ✗ |
| 9 | 1 | [put the wine bottle on the rack](task9_start1.mp4) | ✗ | ✗ | ✗ | ✗ | ✗ |

| Condition | Successes |
|---|---|
| Instruction only | 15/20 |
| Goal: demo end | 13/20 |
| Goal: demo 4 s ahead | 16/20 |
| Goal: ImageWAM | 13/20 |
| Goal: wrong task | 0/20 |
