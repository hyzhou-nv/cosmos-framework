---
name: cosmos3-nano-sim-transfer-real2real-replace
description: Replace part of a real video — chosen foreground objects, or the background behind them — with Cosmos3 Transfer, holding everything else pixel-stable. Use when the user wants some of a real clip changed and the rest left alone: recolour or re-identify vehicles, swap a ground surface or sky, change a room's walls while people keep moving. Extracts the mask with SAM3 from a text prompt, or takes a mask the user supplies, and runs the 4-step distilled checkpoint with ROI latent pinning.
---

# Cosmos real2real — replace part of a scene

Change the masked region. Hold everything else at the source video's own latents.

This is not restyling the whole frame. Outside the mask the sampler pulls toward the
source at every step, so the untouched part stays close to pixel-identical while the
masked part is regenerated from a control hint and a prompt.

## Scope comes from the user

Change what the user asked to change, and nothing else. An unrequested improvement
elsewhere in the frame defeats the point of this pipeline.

If the user gave a specific request — what to replace, and what it should become — do
that. If they supplied a mask, use it; do not re-derive one.

**If they did not say what to replace, ask before doing anything else.** Use
`AskUserQuestion` with two options:

- **Foreground** — the objects stay in place and change identity or appearance; the
  background is held.
- **Background** — the objects are held; the surface, setting or sky behind them is
  regenerated.

That answer sets the mask polarity and nothing else in the pipeline. Then propose the
specific design — which objects, what they become — and get agreement before running.
Skip that confirmation only when the user delegated the design.

## Invariants

- Source media is read-only. Every attempt writes to a new experiment directory.
- Preserve object count, geometry, pose, motion, action timing, and camera motion. The
  edge hint constrains silhouettes; the prompt must not fight it.
- Inference parameters are fixed by the distilled checkpoint. Steer with the prompt and
  the control hint, never by tuning `guidance`, `num_steps`, `shift` or the sampler.
- The mask means: **white is regenerated, black is pinned to the source.**
- Run only when the user asked for execution.

## Mask polarity

| User wants | `--target` | White (generated) | Black (pinned) |
|---|---|---|---|
| foreground objects replaced | `foreground` | the matched objects | everything else |
| background replaced | `background` | everything else | the matched objects |

`extract_mask.py` always segments the *objects* from the text prompts and then inverts
for `background`. So for background replacement you still name the things to keep.

**Background replacement is the weaker case, and say so when proposing it.** The pipeline
was validated with ~96% of the frame pinned. Inverting that pins ~4%, which constrains
almost nothing — expect the people and objects to stay put while the scene around them
moves freely. If the user wants a background change with the ground geometry held,
include the ground surfaces in the pinned set and replace only the sky or the distance.

## Workflow

Complete [Framework setup](../shared/setup.md) and the
[matched checkpoint step](../inference/SKILL.md#1-select-the-matched-inference-checkpoint),
which exports `SIM_TRANSFER_CHECKPOINT`, `SIM_TRANSFER_TEXT_TOKENIZER` and
`SIM_TRANSFER_VAE`. Then read [references/pipeline.md](references/pipeline.md) for what
this task adds: a second environment for SAM3, and the ROI pin. Put the experiment on
storage every step can see; if the transfer step runs in a container, that means inside
its mount.

Let `S` be this skill's directory, `ROOT` an experiment root, `PY` a Python with numpy,
pillow and imageio-ffmpeg, and `SAM3_PY` the Python of the SAM3 environment. Steps 2 and 6
want a GPU; run them however your site does that — directly, or through a batch job.

**1. Stage the source.** Copies the clip in and decodes PNG frames, because the SAM3 venv
has no video decoder.

```bash
$PY $S/scripts/prep_source.py --video SRC.mp4 --exp-root $ROOT --name r2r_<tag>_<date>
```

**2. Get the mask.** If the user supplied one, drop its frames into
`EXP/inputs/mask_frames/` as `%06d.png`, white where they want change, and skip to 3.
Otherwise run SAM3 with concrete nouns — it is text-grounded, so `"parked car"` works and
`"the cars on the left"` does not.

```bash
SAM3_MODEL=/path/to/sam3 $SAM3_PY $S/scripts/extract_mask.py \
  --frames-dir EXP/inputs/frames --out-dir EXP/inputs/mask_frames \
  --prompt "car" --prompt "person" --target foreground --dilate 4
```

`extract_mask_box.py` takes the same arguments plus one or more `--box x0,y0,x1,y1`, and
keeps only the tracked objects whose centroid on their first frame falls inside a box. Use
it when the noun matches more objects than you want to change. It also propagates backward
from the first detection to fill leading empty frames.

Read `EXP/inputs/mask_summary.json` before going on, and check these three fields in
order:

1. **`frames_with_empty_mask` must be empty.** This is the one that bites, because
   `generated_fraction` is a whole-clip average and hides it: a mask that is all black
   for the opening 37 of 80 frames still reports a plausible-looking 1.1%. Those frames
   pin the entire frame, and since the sampler is causal it then carries the source
   appearance through the rest of the clip — the object comes out unchanged even where
   the mask is open. Measured on two such cases, mask-interior MAE was 6.1 and 6.5
   against 20–104 for every case with a full mask. Never submit a transfer on a mask
   with leading empty frames.
2. **`objects_per_prompt`** must match what is actually in the footage. A prompt that
   matched nothing means the noun is wrong, not that the clip is hard.
3. **`generated_fraction`** should be in the range the design implies.

A noun that is too generic is the usual cause of empty leading frames: `toy block` was
first detected at frame 37, while `red cube` alongside it, with `--min-score 0.25`,
caught the same object from frame 0. Add a colour or material word and lower the score
floor before concluding anything about the clip.

`--dilate` grows the mask to cover contact shadows and matting error — raise it when
object edges come out haloed.

**3. Build the mask video.**

```bash
$PY $S/scripts/build_mask.py --exp EXP
```

Only the mask. The framework derives the edge hint itself from the source, at the geometry
it resolves for the run, so nothing here computes a control map — precomputing one is a
measurably different input, not a shortcut to the same thing. For `seg` or `depth`, which
the framework cannot derive, hand the precomputed video to step 5 as
`--control seg --control-path FILE`.

**4. Write the prompt.** Read [references/prompt-writing.md](references/prompt-writing.md)
first — it covers what this task needs beyond the family's
[prompt guide](../shared/prompt-guide.md) — it is the part of this skill most likely to decide whether the run is any good.
One block of prose at `EXP/prompts/plain.txt`, never JSON. Describe the finished scene;
never write "replace X with Y", never mention the source. Give every replaced object its
own paragraph with its own colour, and describe the pinned surroundings too.

**5. Build the request.**

```bash
$PY $S/scripts/build_request.py --exp EXP --name r2r_v1 --prompt-file EXP/prompts/plain.txt
```

The source goes in as `vision_path` and does two jobs: the framework derives the edge from
it, and the pinned region is held at it. `--edge-preset` picks the Canny preset the
framework applies; `medium` (100/200) is the default and the measured best.

On a first run of a new scene, also build the paired control with `--no-pin --name
r2r_v1_nopin`. Two GPUs run both at once, and the pair is the only reliable evidence that
pinning took effect — see step 7.

**6. Run.**

```bash
REQUESTS=EXP/requests OUT=EXP/generated $S/scripts/run_transfer.sh
```

It reads the three `SIM_TRANSFER_*` variables exported during setup.

It runs every request in `REQUESTS`, one per visible GPU, and skips any that already has
output, so an interrupted wave can simply be re-run. Roughly 3–8 minutes for 80 frames on
one GPU. If the transfer runs behind a different view of the filesystem, pass `REQUESTS`
and `OUT` as that side sees them and build the request with matching
`--repo-root`/`--container-root`. Attach a background waiter; do not poll.

**7. Verify.**

```bash
$PY $S/scripts/verify_run.py --exp EXP --run r2r_v1 --run r2r_v1_nopin
```

It checks frame count, size, FPS and a non-frozen tail, then reports MAE against the
source inside and outside the mask. Outside must be far below inside. If the paired
no-pin run has a similar outside figure, pinning did not take effect — do not report the
run as successful.

## Diagnosing weak output

- **Pinned region drifted** — the `roi_pin_*` fields did not reach the sampler. Compare
  against the no-pin control. Do not attribute this to the prompt.
- **Replaced objects came out white or washed out** — the prompt is structured or thin.
  Give each object its own paragraph, a saturated colour word, and an explicit negation.
- **Wrong body shape** — the class noun does not match the silhouette in the control
  video. Fix the noun, not the parameters.
- **Hue shifted warm-to-cool** — this checkpoint drifts blue toward sky blue and red
  toward magenta. Compensate in the colour word.
- **Haloed object edges** — raise `--dilate`.
- **Objects blobby, no structure** — segmentation control carries no outline information
  and a 4-step budget cannot recover it. Use edge.
- **Object missing from the mask** — the SAM3 noun is wrong, or `--min-score` is too
  high for a small or occluded instance. A noun that is too broad is the commoner
  failure: `"folded clothing on a bed"` matched the whole duvet (18.9% of the frame),
  while `"crumpled red cloth"` matched just the garment (1.3%).
- **Masked object came out identical to the source** — check
  `frames_with_empty_mask` first, before touching the prompt. Leading empty frames
  produce exactly this symptom and the mask coverage looks normal.
- **Only one of several matched objects should change** — SAM3 cannot take a box
  prompt (the video session API is text-only), so track the noun and then keep only the
  object whose centroid falls inside a box you choose. One cube out of three was
  isolated this way.

## Known limitation: adjacent objects of the same material

Several same-material objects touching each other do not reliably come out as separate
objects with separate colours. Three folded garments in one tray, prompted as three
towels in three colours with explicit left/middle/right ordering, rendered as two colours.

Three levers were measured on that scene and none of them fixed it. The preset flag now
lives on `build_request.py`, and these numbers were measured with a precomputed edge map;
the direction holds, the exact figures were not re-measured against the
framework-derived one:

| attempt | result |
|---|---|
| `--keep-instances-apart` seam in the mask | no effect — a 1–2px seam does not survive the reduction to latent cells |
| `--edge-preset low` | **worse**: all three merged into one colour, mask-interior MAE 45.4 → 78.3 |
| `--edge-preset very_low` | **worse**: one colour, MAE 45.4 → 82.4 |

Lowering the edge threshold adds fabric wrinkles, not object boundaries, and the extra
edges make the model read the region as one continuous crumpled surface. `medium` stays
the default; do not lower it to separate objects. The pinned region was unaffected across
all three (outside-mask MAE 3.32 / 3.34 / 3.45), so this is a binding limit inside the
mask, not a pinning problem.

If separate identities matter, mask and run the objects one at a time.

Do not reach for 50 steps. Measured on the same scene it was worse outside the mask than
4-step with pinning, and took 30× longer.
