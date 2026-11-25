# π0.7 reference subgoals vs base BAGEL

Goal: feed base BAGEL (DFloat11, zero-shot) the same inputs π0.7's world model received, and compare its subgoals with π0.7's. π0.7's world model is BAGEL fine-tuned on their robot data, so the gap shows how much that fine-tuning contributes.

## Inputs (collected 2026-10-28)

- `current_1..5.png`, `subgoal_1..5.png`: from the [π0.7 project page](https://pi.website/pi07), where they're labelled "Current observation 1–5" and "Subgoal 1–5". About 240×171 (observations) and 212×150 (subgoals), fisheye front camera of a mobile robot.
- `paper_fig3/`: all images on page 6 of the paper (Fig. 3, "Prompt overview"), extracted with `pdfimages`. Odd-numbered files are blank transparency masks.

**Pairing.** The website pairs are the five columns of the paper's "put food on table" example (Fig. 3). Each `subgoal_i` matched its Fig. 3 column by pixel similarity, and the subtask text comes from that figure:

| Step | Subtask (Fig. 3) | Kind of change |
|---|---|---|
| 1 | push the open button on the microwave | Edit of the same view |
| 2 | pick up the plate of food in the microwave | Edit of the same view |
| 3 | move to the dining table | Location change (robot drives) |
| 4 | put the plate with food on the dining table | Edit of the same view |
| 5 | close the microwave | Location change (back to the kitchen) |

π0.7's metadata for this task was "Quality: 5/5, Mistake: false".

The Fig. 3 shirt-folding task (UR5e) isn't used yet. It has no subtask text, and its observation/subgoal pairing couldn't be matched reliably at thumbnail size.

## Test

`cases.json` has 10 cases: each step with two prompts.

- `bare`: the subtask text exactly as π0.7 used it
- `template`: the LIBERO probe's editing instruction wrapped around the subtask

The run uses `pi07_code/probe_bagel.py` with the GPU-resident model (`--cpu-offload-blocks 0`), seed 0 and BAGEL's image-editing settings. Results go to `results_bagel/`, with the log in `run.log`.

## Results (run 2026-10-28, seed 0, model fully on GPU)

About 67 s per image at 704×512, 23.2 GB peak GPU memory. The compact view is `results_bagel/summary_grid.png`: input, π0.7's subgoal, base BAGEL with the bare prompt, base BAGEL with the template.

| Step | π0.7 (fine-tuned BAGEL) | Base BAGEL, bare subtask | Base BAGEL, template prompt |
|---|---|---|---|
| 1. Open the microwave (edit) | Door open, arm at the door | No change | No change |
| 2. Pick up the plate (edit) | Door open, gripper at the plate | Pasted a plate of food floating outside a closed microwave | Pasted a plate on the counter, plus an invented black object |
| 3. Move to the dining table (location) | The real dining area | Invented an empty purple room | Invented a different dining room |
| 4. Put the plate on the table (edit) | Plate set down, arm pulled back | Added a second dish of food | Almost no change |
| 5. Close the microwave (location) | Back in the kitchen, door closed | Drew a small microwave on the dining table | Drew a black box on the table |

**Verdict: 0/5 with either prompt.** The failure modes match the LIBERO test (`../bagel_subgoal_probe/`, 0/3): no physical change, the named object drawn instead of the action, and invented scenes. Prompt wording isn't the bottleneck. The robot fine-tuning in π0.7's recipe is what makes its world model work. One seed only; a second seed wouldn't change a 0/5 result of this kind.

## Reading the results

- In `results_bagel/comparison.png`, the third column is titled "Later demo frame (reference only)" by the script. Here it is **π0.7's generated subgoal**, not a real future frame.
- Steps 3 and 5 need knowledge of another room that zero-shot BAGEL can't have. Judge editing ability on steps 1, 2 and 4.
- Inputs are small fisheye thumbnails, upscaled by BAGEL to 512 px. That handicaps both the input and the comparison.
