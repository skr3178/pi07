# π0.7 equations

Source: [../pi_07.pdf](../pi_07.pdf) (π0.7, Physical Intelligence, April 2026). Every equation and math definition in the paper is below, with its section and page.

The paper has only one numbered equation (Eq. 1) and one algorithm (Algorithm 1); the other formulas are written inline. Sections 1–7 and 9 cover the paper itself. The rest is clearly marked as not from the paper:

- Section 8 adds formulas an implementation needs that the paper leaves out.
- Section 10 writes out as pseudocode the procedures the paper describes only in prose.
- Section 11 gives our toy versions (LIBERO + SmolVLA).

## 1. Notation (Sec. III, p. 3; Sec. V, pp. 4–5)

Observation at time $t$: $n$ camera images plus the robot's joint configuration.

$$
\mathbf{o}_t = [\mathbf{I}_t^1, \ldots, \mathbf{I}_t^n, \mathbf{q}_t]
$$

| Symbol | Meaning |
|---|---|
| $\mathcal{D}$ | Training dataset of robot trajectories |
| $\mathbf{a}_t$ | Action at time $t$ (joint or end-effector command) |
| $\mathbf{a}_{t:t+H}$ | Action chunk: the next $H$ actions |
| $\mathbf{o}_{t-T:t}$ | Recent observation history |
| $\hat{H} < H$ | Number of chunk steps actually executed |
| $\mathcal{C}_t$ | Prompt / context. Conventionally just a language instruction: $\mathcal{C}_t = (\ell_t)$ |
| $\ell_t$ | Task instruction, e.g. "clean the kitchen" |
| $\hat{\ell}_t$ | Subtask instruction, e.g. "open the fridge door" |
| $m$ | Episode metadata (speed, quality, mistake) |
| $c \in \{\texttt{joint}, \texttt{ee}\}$ | Control mode |
| $\pi_\theta$ | The VLA policy |
| $g_\psi$ (also written $p_\psi$) | The world model that generates subgoal images |
| $\mathcal{D}_g$ | Subset of segments with high-quality subtask labels, used to train the world model |

Multi-view subgoal images, where $\mathbf{G}_t^i$ is the desired near-future image for camera $i$:

$$
\mathbf{g}_t = [\mathbf{G}_t^1, \ldots, \mathbf{G}_t^n]
$$

Ground-truth subgoal: the frame at the end of the segment that contains $t$.

$$
\mathbf{g}_t^\star = \mathbf{o}_{t_{\text{end}}}
$$

Full context used at test time (Algorithm 1):

$$
\mathcal{C} = \{\ell,\ \hat{\ell},\ \mathbf{g}^\star,\ m,\ c\}
$$

## 2. VLA training objective (Eq. 1, Sec. III, p. 3)

$$
\max_\theta \; \mathbb{E}_{\mathcal{D}} \left[ \log \pi_\theta\left(\mathbf{a}_{t:t+H} \mid \mathbf{o}_{t-T:t}, \mathcal{C}_t\right) \right] \tag{1}
$$

The paper notes that a flow-matching action expert optimizes an approximate lower bound on this log-likelihood, not the closed form. Section 8 gives the actual loss.

## 3. World model objective (Sec. V-B, p. 5)

$$
\max_\psi \; \mathbb{E}_{\mathcal{D}_g} \left[ \mathcal{L}_{\text{CFM}}\left( \mathbf{g}_t^\star,\ g_\psi\left(\mathbf{o}_t, \hat{\ell}_t, m\right) \right) \right]
$$

- $\mathcal{L}_{\text{CFM}}$ is the standard conditional flow-matching loss (the paper cites Lipman et al. and doesn't write it out).
- The target is the end-of-segment frame, $\mathbf{g}_t^\star = \mathbf{o}_{t_{\text{end}}}$.
- The model is conditioned on the current observation, the subtask instruction and the episode metadata.

## 4. Test-time prompting (Algorithm 1, p. 7)

Sample subgoal images from the world model:

$$
\mathbf{g}^\star \sim p_\psi\left(\mathbf{g}^\star \mid \mathbf{o}_t, \hat{\ell}, m\right)
$$

Sample an action chunk from the VLA:

$$
\mathbf{a}_{t:t+H} \sim \pi_\theta\left(\mathbf{a} \mid \mathbf{o}_{t-T:t}, \mathcal{C}\right)
$$

With real-time chunking, the chunk is also conditioned on the already-committed actions $\mathbf{a}_{t:}$:

$$
\mathbf{a}_{t:t+H} \sim \pi_\theta\left(\mathbf{a} \mid \mathbf{o}_{t-T:t}, \mathcal{C}, \mathbf{a}_{t:}\right)
$$

Algorithm 1, transcribed:

```
Input: initial observation o_0, task instruction l, episode metadata m, control mode c
 1  initialize subtask l^ (from high-level policy or coaching)
 2  g* ~ p_psi(g* | o_0, l^, m)
 3  C = {l, l^, g*, m, c}
 4  a_{t:t+H} ~ pi_theta(a | o_{t-T:t}, C)                  # optional: CFG
 5  for t = 0, 1, 2, ... do
 6      if l^ changed or Delta-second timer elapsed then
 7          g* ~ p_psi(g* | o_t, l^, m)                     # non-blocking (async)
 8          C = {l, l^, g*, m, c}
 9      end if
10      if H^ steps elapsed since last inference then
11          a_{t:t+H} ~ pi_theta(a | o_{t-T:t}, C, a_{t:})  # async with RTC
12      end if
13      execute a_t
14  end for
```

## 5. Classifier-free guidance on the prompt (Sec. VII, p. 7)

Each action denoising step follows:

$$
\nabla_{\mathbf{a}} \log \pi_\theta\left(\mathbf{a}_{t:t+H} \mid \mathbf{o}_t, \mathcal{C}_t\right)
+ \beta \left( \nabla_{\mathbf{a}} \log \pi_\theta\left(\mathbf{a}_{t:t+H} \mid \mathbf{o}_t, \mathcal{C}_t\right)
- \nabla_{\mathbf{a}} \log \pi_\theta\left(\mathbf{a}_{t:t+H} \mid \mathbf{o}_t, \mathcal{C}_t^{\text{uncond}}\right) \right)
$$

- $\mathcal{C}_t^{\text{uncond}}$ is the context in "unconditional" mode, with the guided part dropped.
- $\beta$ is the CFG weight, and the paper uses $\beta \in \{1.3,\ 1.7,\ 2.2\}$.
- Any prompt part could be guided, but the paper applies CFG to the episode metadata.
- The positive and negative prompts are packed into one sequence as two branches that don't attend to each other (Fig. 19).

## 6. Numeric settings stated in the paper

| Quantity | Value | Where |
|---|---|---|
| Chunk length $H$ | 50 action steps | Sec. VI-B |
| Executed steps $\hat{H}$ | $\hat{H} \in \{15, 25\}$ | Sec. VII |
| Denoising steps | 5 | Sec. VII |
| Subgoal refresh interval $\Delta$ | 4 s (matches SuSIE) | Sec. VII, App. C |
| Simulated RTC delay | 0–12 timesteps, i.e. at most 12 / (50 Hz) = 240 ms | Sec. VI-B |
| Observation history | Up to 6 frames per camera, 1 s stride | Sec. VI-B |
| VLA image size | 448 × 448 | Sec. VI-B |
| World model ViT input | 448 × 336, patch 14 | App. C |
| World model VAE input and target | 512 × 384, patch 16 | App. C |
| Speed metadata | Episode length binned every 500 steps (1750–2250 → "2000 steps") | Sec. V-C |
| Quality metadata | Integer 1–5 (5 = best) | Sec. V-C |
| Mistake metadata | true / false per action segment | Sec. V-C |
| Test-time speed | 15th percentile of the task's episode lengths | Sec. VII |
| Test-time quality, mistake | 5, false | Sec. VII |
| CFG weight $\beta$ | 1.3, 1.7 or 2.2 | Sec. VII |
| Parameter counts | 4B VLM (incl. 400M vision encoder) + 860M action expert ≈ 5B; world model 14B | Sec. IV, V-B |

## 7. Training-time sampling probabilities (Sec. V-E, VI-B, VI-C)

| Event | Probability |
|---|---|
| Subgoal images included in an example | $P = 0.25$ |
| Real subgoal = end-of-segment frame | $P = 0.25$ |
| Real subgoal = frame $t' \sim \mathcal{U}[t,\ t + 4\,\text{s}]$ | $P = 0.75$ |
| Subtask $\hat{\ell}_t$ dropped, given a subgoal is present | $P = 0.30$ |
| All metadata $m$ dropped | $P = 0.15$ |
| Each metadata field dropped individually | $P = 0.05$ |
| Whole observation history dropped | $P = 0.30$ |
| Rear-view camera dropped | $P = 0.30$ |
| Control mode $c$ dropped | never |

Besides these real frames, many extra training examples use world-model subgoals in place of real future frames (Sec. VI-C). The paper gives no ratio.

Derived (not in the paper): a given metadata field is present with probability $(1 - 0.15)(1 - 0.05) = 0.8075$.

## 8. Formulas the paper leaves out (not from the paper)

**Flow-matching loss for the action expert.** This is the π0/π0.5 convention, read from the local LeRobot port (`~/Downloads/lerobot/src/lerobot/policies/pi05/modeling_pi05.py:678-734`). π0.7 builds on that architecture, but the paper doesn't confirm it keeps this exact form.

$$
\mathbf{x}_\tau = \tau\,\boldsymbol{\epsilon} + (1 - \tau)\,\mathbf{a}_{t:t+H}, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

$$
\mathbf{u} = \boldsymbol{\epsilon} - \mathbf{a}_{t:t+H}
$$

$$
\mathcal{L}(\theta) = \mathbb{E}_{\tau, \boldsymbol{\epsilon}, \mathcal{D}} \left\| \mathbf{v}_\theta\left(\mathbf{x}_\tau, \tau \mid \mathbf{o}_{t-T:t}, \mathcal{C}_t\right) - \mathbf{u} \right\|^2
$$

$$
\tau = 0.999\,s + 0.001, \qquad s \sim \mathrm{Beta}(1.5,\ 1)
$$

**Sampling (Euler integration from noise to actions).** Start from pure noise at $\tau = 1$ and step toward $\tau = 0$ in $N$ steps; the paper uses $N = 5$.

$$
\mathbf{x}_1 = \boldsymbol{\epsilon}, \qquad \mathbf{x}_{\tau + \delta} = \mathbf{x}_\tau + \delta\, \mathbf{v}_\theta(\mathbf{x}_\tau, \tau \mid \mathbf{o}, \mathcal{C}), \qquad \delta = -\tfrac{1}{N}
$$

**CFG in velocity form.** The paper states CFG with scores. Code applies it to the predicted velocity, which is the usual practice because the velocity of a Gaussian flow path is an affine function of the score.

$$
\mathbf{v} = \mathbf{v}_{\text{cond}} + \beta\left(\mathbf{v}_{\text{cond}} - \mathbf{v}_{\text{uncond}}\right) = (1 + \beta)\,\mathbf{v}_{\text{cond}} - \beta\,\mathbf{v}_{\text{uncond}}
$$

**World model token grids.** App. C says the two input sizes differ because of the different patch sizes. Both give the same 32 × 24 grid:

$$
\tfrac{448}{14} \times \tfrac{336}{14} = 32 \times 24 = 768 \text{ ViT tokens}, \qquad \tfrac{512}{16} \times \tfrac{384}{16} = 32 \times 24 = 768 \text{ VAE tokens}
$$

## 9. Inconsistencies in the paper's notation

- **The world model has two names.** It is $g_\psi$ in Sec. V-B and $p_\psi$ in Algorithm 1.
- **"max" of a loss.** The world model objective takes $\max_\psi$ of $\mathcal{L}_{\text{CFM}}$. As a loss, it should be minimized.
- **Conditioning on history.** Eq. 1 and Algorithm 1 condition on the history $\mathbf{o}_{t-T:t}$, while the CFG formula writes only $\mathbf{o}_t$.
- **Subgoal token position.** Sec. VI-B and the example prompt put subgoal images before the text. The Fig. 19 caption says image goals are an extra block "after the text prompt".

## 10. Procedures described only in prose (reconstructed, not from the paper)

The paper describes these steps in words. The pseudocode follows the cited sections; anything the paper doesn't state is marked "not stated".

### 10.1 Building one policy training example (Sec. V-E, VI-B, VI-C)

```
Input: trajectory o_0..o_L with actions; task l; segments k with subtask l^_k over [s_k, e_k];
       metadata m; control mode c

1  pick a time t inside some segment k
2  observation: cameras at t, plus history frames 1 s apart (up to 6)
     drop the whole history with p = 0.30; drop the rear camera with p = 0.30
3  target: action chunk a_{t:t+H}, H = 50
4  with p = 0.25, add subgoal images g:
       with p = 0.25:  g = o_{e_k}                (end of the segment)
       otherwise:      g = o_{t'}, t' ~ U[t, t + 4 s]
                       (clipping at the segment or episode end: not stated)
     some extra examples use world-model images instead of real frames (ratio: not stated)
     if g was added: drop the subtask text l^_k with p = 0.30
5  metadata: drop all of m with p = 0.15; otherwise drop each field with p = 0.05
6  control mode c: never dropped
7  training-time RTC: simulate an inference delay of 0-12 steps
     (how the delay conditions the chunk follows refs [107, 108]; not detailed in this paper)
8  losses: flow matching on the action expert; cross-entropy on FAST action tokens for the
     backbone; expert gradients don't reach the backbone (knowledge insulation)
```

### 10.2 Building one world-model training example (Sec. V-B, App. C)

```
Input: segment k with an especially high-quality subtask label l^_k over [s_k, e_k]; metadata m

1  pick a time t inside the segment
2  inputs:  3 current camera images o_t (encoded twice: ViT at 448x336, VAE at 512x384),
            subtask l^_k, metadata m
3  target:  the 3 camera images at the segment end, o_{e_k} (VAE at 512x384)
4  loss:    flow matching, following BAGEL's training recipe
5  data mix: robot and egocentric human segments, plus open-source image-editing and video
            datasets (mixing ratios: not stated)
```

### 10.3 One action-denoising pass with CFG (Sec. VII, Fig. 19)

```
pack one sequence with two branches that cannot attend to each other:
   branch +: the full context C_t
   branch -: C_t with the guided part removed (the paper guides on metadata)
start from noise at tau = 1; for each of 5 denoising steps:
   v+, v- = action-expert outputs for the two branches
   v      = v+ + beta (v+ - v-),   beta in {1.3, 1.7, 2.2}
   take one Euler step with v
```

The paper writes the guidance with scores; the velocity form is in section 8. The world model uses the same packing trick with three guidance groups instead of two (App. C); its weights aren't stated.

### 10.4 Attention rules (Sec. VI-B, Fig. 19)

| A token in… | …can attend to | Source |
|---|---|---|
| Observation (incl. history) | Other observation tokens (bidirectional) | Sec. VI-B |
| Subgoal images | Each other (bidirectional) and the observation | Sec. VI-B |
| Text (task, subtask, metadata) | Earlier tokens, causally | Sec. VI-B. The subgoal-vs-text order conflicts with Fig. 19; see section 9 |
| FAST action tokens (training only) | The prefix; **not** the flow-matching action tokens | Fig. 19 |
| Flow-matching action tokens | Each other (bidirectional) and all backbone activations; **not** the FAST tokens | Sec. VI-B, Fig. 19 |
| CFG branches | Never each other | Fig. 19 |
| State tokens | Not stated | – |

Without subgoal images, the pattern falls back to π0.5's: global bidirectional attention among all image views (Fig. 19).

## 11. Our toy versions (LIBERO + SmolVLA; not from the paper)

### 11.1 Policy loss and sampling

Read from `lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py` (lines 518–531, 674–737). It's the same convention as section 8:

$$
\mathbf{x}_\tau = \tau\,\boldsymbol{\epsilon} + (1-\tau)\,\mathbf{a}_{t:t+H}, \qquad \mathbf{u} = \boldsymbol{\epsilon} - \mathbf{a}_{t:t+H}, \qquad \tau = 0.999\,s + 0.001,\; s \sim \mathrm{Beta}(1.5,\ 1)
$$

$$
\mathcal{L}(\theta) = \mathbb{E}\left[\, \mathbb{1}_{\text{in episode}} \cdot \left\| \mathbf{v}_\theta(\mathbf{x}_\tau, \tau \mid \mathbf{o}_t, \mathbf{q}_t, \text{prompt}, \mathbf{g}) - \mathbf{u} \right\|^2 \right]
$$

- **Mask:** chunk steps past the episode end are masked out of the loss.
- **Shapes:** LIBERO's 7-D actions are padded to 32 dims (`max_action_dim`); chunk length $H = 50$, kept from `smolvla_libero`.
- **Sampling:** $N = 10$ Euler steps by default (`num_steps`; the paper uses 5), from $\tau = 1$ with $\delta = -1/N$.
- **Differences from π0.7:** no FAST-token loss (a frozen backbone stands in for knowledge insulation), no RTC, no history, and a single current frame $\mathbf{o}_t$ instead of $\mathbf{o}_{t-T:t}$.

### 11.2 Prompt and its length

The prompt follows the paper's example format, without the control-mode field:

```
Task: {l}. Subtask: {l^}. Speed: {speed bin}. Quality: {q}. Mistake: {true|false}.
```

**SmolVLA cuts prompts at 48 tokens** (`tokenizer_max_length = 48` in `configuration_smolvla.py`). Long LIBERO-10 tasks plus subtask and metadata will exceed that, and truncation drops the metadata at the end. Measure the longest prompt with the SmolVLM2 tokenizer and raise the limit (e.g. to 96) for both arms A and B.

### 11.3 World-model objective

Training pairs come from `libero_10_subtask`'s verified segments (`dataset.md`): input = scene-camera frame $\mathbf{o}_t$ and subtask text $\hat{\ell}_k$; target = the scene-camera frame at the segment end, $\mathbf{o}_{e_k}$. Drop task 3's 1-frame id-15 segments. Task 9 has a single segment, so its target is the episode end.

The loss depends on the model chosen (`model.md`):

- **ImageWAM / FLUX.2:** flow matching, the same family as the paper's BAGEL objective (section 3).
- **InstructPix2Pix:** the standard noise-prediction diffusion loss, where $\mathcal{E}$ is the VAE encoder and $\bar\alpha_\tau$ the noise schedule:

$$
\mathcal{L}(\phi) = \mathbb{E}\left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\phi\!\left(\mathbf{z}_\tau, \tau \mid \mathcal{E}(\mathbf{o}_t), \hat{\ell}_k\right) \right\|^2, \qquad \mathbf{z}_\tau = \sqrt{\bar\alpha_\tau}\, \mathcal{E}(\mathbf{o}_{e_k}) + \sqrt{1-\bar\alpha_\tau}\, \boldsymbol{\epsilon}
$$

### 11.4 Horizons and probabilities in steps (10 fps)

| Quantity | Paper | Toy |
|---|---|---|
| Subgoal present | $p = 0.25$ | Same |
| Real subgoal frame | End of segment ($p = 0.25$) or $t' \sim \mathcal{U}[t, t + 4\,\text{s}]$ | End of segment ($p = 0.25$) or $t' \sim \mathcal{U}[t, t + 40]$ steps, clipped to the episode end |
| Subgoal refresh $\Delta$ | 4 s | 40 steps |
| Subtask dropped with a subgoal | $p = 0.30$ | Same |
| Metadata dropout | 15% whole + 5% per field | Same |
| Speed metadata | Episode length, 500-step bins at 50 Hz | Episode length binned within task; bin width to choose (LIBERO-10 medians are 187–409 steps) |
| Quality, mistake | Human labels | From the rubric declared in `dataset.md` (not yet defined); unlabelled demos get an explicit "unknown" |
| History, rear camera, RTC delay | 30%, 30%, 0–12 steps | None (not used) |

### 11.5 Algorithm 1, toy version (synchronous LIBERO loop)

```
Input: task l, desired metadata m* (e.g. Quality 5, Mistake false, fast Speed), initial obs o_0

 1  l^ <- first subtask of the task
       (v1: oracle switching from the simulator's goal predicates; this stands in for
        pi0.7's high-level policy and is privileged information)
 2  g  <- W(scene image of o_0, l^)                        # world model; arm B only
 3  for t = 0, 1, 2, ... until success or the env's step limit:
 4      if l^ changed or 40 steps since g was made:  g <- W(scene image of o_t, l^)   # arm B only
 5      if the action queue is empty:
 6          a_{t:t+50} <- SmolVLA(o_t, q_t, prompt(l, l^, m*), g)    # optional CFG on m*
 7          queue the first H^ actions (H^ to tune, e.g. 10-25)
 8      execute the next queued action; observe o_{t+1}
```

Arm A runs the same loop without lines 2 and 4, with a task-only prompt and no subgoal. Everything runs synchronously, so world-model latency slows wall-clock time, not the simulated robot.
