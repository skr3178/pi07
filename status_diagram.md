# π0.7 toy: status at a glance (2026-10-30)

Snapshot of the goal-image investigation. Details: [todo.md](todo.md).

## 1. What we're building (π0.7 Section V, toy version)

```
                 ┌────────────────────────── prompt ───────────────────────────┐
  task text ───► │ "put soup in basket"  + subtask text  + metadata  + control │
                 └─────────────────────────────────────────────────────────────┘
                                              │
  camera now ──────────────┐                  ▼
                           ├────────►  ┌──────────────┐      ┌───────────────┐
  GOAL IMAGE (≈1 s ahead) ─┘           │   SmolVLA    │ ───► │ 50 robot      │
     ▲                                 │   policy     │      │ actions       │
     │                                 └──────────────┘      └───────────────┘
  ┌──┴───────────┐
  │  ImageWAM    │  world model: "what should the scene look like next?"
  │  (tested ✓)  │  follows LIBERO-Goal instructions ✓
  └──────────────┘  on LIBERO-10 does the usual next step, not the instructed one ✗
```

## 2. The key experiment: does a goal image help the policy?

```
   Arm A:  camera + text ─────────────► SmolVLA ─► actions
   Arm B:  camera + text + GOAL IMG ──► SmolVLA ─► actions

   Paper's claim:  B should learn faster and make smaller errors than A
   Our result:     B is WORSE than A  ✗
```

## 3. How we narrowed it down (lower = better, first-8 action error)

```
 smolvla_libero, untouched ............ 0.123  ◄── already very good (!)
 │
 ├─ fine-tune A, lr 1e-4 .............. 0.289  ✗ training damaged it
 ├─ fine-tune A, lr 1e-5 .............. 0.227  ✗ less damage, still worse
 │
 └─ B with goal images (every fix tried):
      original code ............ 0.323  ✗
      + marker + 4 layers ...... 0.322  ✗   ┐
      isolated layout .......... 0.309  ✗   ├─ all worse than A (0.228)
      (B with goal turned off) . 0.216      ┘

 Ruled out:  ✓ attention layout   ✓ goal frames aligned (10/10)
             ✓ normalization      ✓ marker / unfreezing   ✓ text contamination
```

## 4. Diagnosis and next step

```
   smolvla_libero was trained on THESE SAME LIBERO-10 demos
            │
            ▼
   It already knows the answer from the camera alone
            │
            ▼
   A goal image adds nothing → just a distraction → B gets worse

   ┌──────────────────────────── NEXT TEST ──────────────────────────────────┐
   │ Start both arms from smolvla_base (has never seen LIBERO)               │
   │                                                                         │
   │   error │\  A                         if B drops faster → goal pathway  │
   │         │ \___                          works, so use smolvla_base for  │
   │         │\    \_____                    the real A/B                    │
   │         │ \__ B     ‾‾‾‾                                                │
   │         │    ‾‾‾‾‾‾‾‾‾‾                if not → the problem is in how   │
   │         └──────────────► updates        SmolVLA handles goal images     │
   │                                                                         │
   │ Cost: ~45–60 min GPU. smolvla_base downloaded (873 MB).                 │
   └─────────────────────────────────────────────────────────────────────────┘
```

## 5. Progress overview

```
 [✓] envs built (ImageWAM, LeRobot+LIBERO)    [✓] ImageWAM goal-image tests
 [✓] A/B training + closed loop (Codex, 0/6)  [✓] goal-pathway fixes (none helped)
 [✓] smolvla_base downloaded (/media/skr/storage/models/smolvla_base)
 [ ] smolvla_base A/B test   ◄── you are here
 [ ] real A/B with 25% goals + Section V prompt (text/metadata/control dropout)
```
