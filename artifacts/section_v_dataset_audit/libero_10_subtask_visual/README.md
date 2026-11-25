# libero_10_subtask: subtask id meanings (visual check)

Checked 2026-10-28 on the local copy at `data/libero_10_subtask` (500 episodes, 138,090 frames).

**Method.**

- **Order:** read the `subtask_index` sequence of every episode and counted the orderings per task.
- **Frames:** for each id, extracted the start and end frames of 3 segments from the scene camera (`observation.images.image`). These are the two contact sheets in this folder; each row is one id, S = segment start, E = segment end.
- **Reproduce:** `python subtask_frames.py <dataset root> <output dir>`

**Evidence column.** "Frames" means the start→end change is visible in the contact sheet. "Shared id" means the item's identity comes from which tasks share the id, consistent with the frames: at 160 px the basket items move into the basket, but their labels can't be read.

## Id → subtask

| Id | Subtask | Tasks | Segments | Median length (frames) | Evidence |
|---|---|---|---|---|---|
| 0 | pick up the book and place it in the back compartment of the caddy (whole task) | 9 | 1 | 165 | Frames; one episode only, same content as id 10 |
| 1 | close the microwave | 2 | 50 | 133 | Frames: door open → closed |
| 2 | close the drawer | 1 | 50 | 99 | Frames: drawer open → closed |
| 3 | put the chocolate pudding to the right of the plate | 8 | 50 | 127 | Frames, plus shared id (3 is task 8's non-mug step) |
| 4 | put the moka pot on the stove | 0, 3 | 98 | 200 | Frames: pot moves onto the burner. In task 3 it's the second pot. |
| 5 | put the first moka pot on the stove | 3 | 49 | 182 | Frames: two pots on the table → one on the burner |
| 6 | put the white mug on the plate | 7, 8 | 100 | 112 | Frames, plus shared id (white mug is common to tasks 7 and 8) |
| 7 | put the yellow and white mug on the right plate | 7 | 50 | 149 | Frames |
| 8 | put the black bowl in the bottom drawer | 1 | 50 | 144 | Frames: bowl on table → robot placing it in the open drawer |
| 9 | put the alphabet soup in the basket | 4, 5 | 100 | 136 | Shared id (only item common to tasks 4 and 5) |
| 10 | pick up the book and place it in the back compartment of the caddy (whole task) | 9 | 49 | 187 | Frames: book on table → in the caddy |
| 11 | put the butter in the basket | 6 | 50 | 119 | Shared id (task 6's non-cream-cheese item) |
| 12 | put the cream cheese box in the basket | 4, 6 | 100 | 137 | Shared id (only item common to tasks 4 and 6) |
| 13 | put the tomato sauce in the basket | 5 | 50 | 144 | Shared id (task 5's non-soup item) |
| 14 | put the yellow and white mug in the microwave | 2 | 50 | 155 | Frames: mug on table → robot placing it in the open microwave |
| 15 | turn on the stove | 0 (real); 3 (artifact) | 100 | 86 in task 0; 1 in task 3 | Frames: burner dark → red. In task 3 it is always a single frame at the episode start: a labelling artifact, not a step. |

## Order per task (all 500 episodes)

| Task | Instruction | Sequence (episodes) |
|---|---|---|
| 0 | turn on the stove and put the moka pot on it | 15 → 4 (50) |
| 1 | put the black bowl in the bottom drawer of the cabinet and close it | 8 → 2 (50) |
| 2 | put the yellow and white mug in the microwave and close it | 14 → 1 (50) |
| 3 | put both moka pots on the stove | 15 → 5 → 4 (47); 15 → 5 (2); 15 → 4 (1). Id 15 here is the 1-frame artifact. |
| 4 | put both the alphabet soup and the cream cheese box in the basket | 9 → 12 (50) |
| 5 | put both the alphabet soup and the tomato sauce in the basket | 9 → 13 (50) |
| 6 | put both the cream cheese box and the butter in the basket | 12 → 11 (50) |
| 7 | put the white mug on the left plate and put the yellow and white mug on the right plate | 6 → 7 (50) |
| 8 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 6 → 3 (49); 3 → 6 (1) |
| 9 | pick up the book and place it in the back compartment of the caddy | 10 (49); 0 (1). Single segment, no split. |

## Caveats for use

- **Drop the 1-frame id 15 segments in task 3** before using segment endpoints as subgoal targets.
- **Task 9 has no subtask split,** so it can't test subtask conditioning. Its segment endpoint is the episode end.
- **Three task-3 episodes are incomplete sequences** (15 → 5, 15 → 4), and one task-8 episode is reversed (3 → 6). Check them before trusting their endpoints.
- **Ids 9, 11, 12 and 13 rest on shared-id logic.** A higher-resolution look (or the wrist camera) would confirm item identity.
- **Segment boundaries weren't checked frame by frame,** only start and end frames for 3 samples per id.
