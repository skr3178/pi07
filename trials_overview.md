# Trials overview (2026-11-25, corrected)

Every trial so far (left) and the proposed next step (right). Details: [todo.md](todo.md), [plan_2026-11-25.md](plan_2026-11-25.md), [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md).

```
ALL TRIALS SO FAR   ✓ worked  ✗ no benefit  ≈ tie  • reference  – not run        ┌─ MY PROPOSAL ──────────────────────┐
(policy: held-out first-8 action error, lower = better · live = sim successes)   │ NEXT: spatial subgoal test         │
                                                                                 │                                    │
A. SUBGOAL IMAGE GENERATORS (world model)                                        │ Goal as 8 numbers, not a picture:  │
 1 BAGEL base, zero-shot       LIBERO 0/3 · π0.7 site 0/5         ✗              │  gripper x,y,z + rotation +        │
 2 ImageWAM, LIBERO frames     plausible, ~0.8 s ahead            ✓              │  finger opening at the target      │
 3 ImageWAM, π0.7 site images  mostly copies the input            ✗              │ → skips the frozen vision          │
 4 ImageWAM, opposing prompts  text changes image less than seed  ✗              │   encoder, the main suspect        │
 5 ImageWAM, LIBERO-Goal       59/60 match the instruction        ✓              │                                    │
 6 ImageWAM, LIBERO-10 steps   habitual step, not instructed      ✗              │ 2 arms vs saved A (0.421):         │
                                                                                 │  S-near  state 8–16 frames ahead   │
B. POLICY FROM smolvla_libero  (untouched: 0.123)                                │  S-end   state at subtask end      │
 7 untouched, live             LIBERO-Goal 6/6 · LIBERO-10 1/6    •              │ checks: real · copy · wrong · off  │
 8 Codex pilot, 200 updates    A 0.404 · B real 0.518             ✗              │  + episode-grouped 95% CI          │
 9 Codex A/B, 25% goals        A 0.289 · B real 0.340             ✗              │ cost: ~30 min code, ~30 min GPU    │
10 Codex live, 6 starts        A 0/6 · B off 0/6 · ImageWAM 0/6   ✗              │                                    │
11 attention-layout check      0.123 in all 3 layouts             ✓              │ ALONGSIDE: encoder probe (minutes) │
12 lr 1e-5                     A 0.227 · B real 0.363             ✗              │  can the frozen encoder's features │
13 goals 100%, 8–16 ahead      B real 0.323 · off 0.227           ✗              │  of current + goal image recover   │
14 + marker + 4 VLM layers     B real 0.322 · off 0.224           ✗              │  how far the gripper moved?        │
15 isolated goal layout        B real 0.309 · off 0.216           ✗              │                                    │
   (9: 1,000 updates at lr 1e-4 · 12–15: 1,000 at lr 1e-5)                       │ IF S clearly beats A:              │
   → likely trained on these demos already; goals only add error                 │  the image pathway is the problem  │
                                                                                 │  → unfreeze last vision blocks,    │
C. NEW STARTING CHECKPOINT                                                       │    or feed goal − current          │
16 smolvla_base                wrong robot (SO100); not used      –              │                                    │
17 smolvla_libero_90           untouched 0.551: room to learn     ✓              │ IF S doesn't help:                 │
                                                                                 │  no goal is usable yet             │
D. POLICY FROM smolvla_libero_90  (2,000 updates, lr 1e-4)                       │  → resume arms to ~10k updates,    │
18 A/B, near goals             A 0.421 · B real 0.417             ≈              │    audit the goal wiring           │
19 3 arms, checkpoints saved   A 0.421 · near 0.410 · end 0.430   ≈              │                                    │
   real ≈ copied-current ≈ wrong-task; all 95% CIs include 0                     │ THEN, each gated on the last:      │
20 Codex live, 20 starts       reference 0/20 · A 2/20 · off 0/20 ≈              │  LIBERO-Goal → subtask text →      │
   (#19's A and B-near)        ImageWAM 4/20 = copy 4/20 (same 4)                │  ImageWAM goals → metadata/CFG     │
                                                                                 │  → final 100-start benchmark       │
E. CODEX SECTION V PIPELINE  (smolvla_libero_90, 1,000 updates)                  │                                    │
21 7 runs: 5 LIBERO-10         goals off beat goals on;           ✗              │ STOP RULE: all variants fail →     │
   profiles + 2 LIBERO-Goal    subtask text ≈ task text                          │  write up the negative result      │
22 live LIBERO-10, 100 eps     task 0/20 · subtask 1/20           ✗              └────────────────────────────────────┘
                               off / copy / ImageWAM 1/20 each
23 checks: gripper-sign fix, CFG math, checkpoint resume          ✓
24 metadata training, 2 runs   interrupted minutes in             –
25 never started: CFG sweep, LIBERO-Goal live test,               –
   500-image ImageWAM cache, final 100-start benchmark

OPEN PROBLEM: no measurable benefit from what a goal image shows,
even when it is a real future frame
```
