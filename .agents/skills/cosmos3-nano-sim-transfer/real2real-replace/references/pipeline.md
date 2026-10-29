# Pipeline and environments

## What ROI latent pinning does

Inside the mask the model generates normally from the control hint. Outside it, the
sampler pulls the predicted velocity toward the source video's latents at every step.

With `x0_pred = x - σ·v`, the velocity that lands the pinned region exactly on the
reference is `v* = (x - x0_ref)/σ`, and the sampler blends

    v ← v + mask · w · (v* - v)

This is deliberately not a hard latent replacement. Writing the clean reference into a
latent that carries noise level σ is a distribution mismatch the model has never seen.
Going through the velocity means the sampler's own re-noising produces
`(1-σ_next)·x0_ref + σ_next·ε` — a correctly noised reference.

The pin is not a control channel. It occupies none of the hint slots, so edge, seg or
depth stays free for the masked region.

The pinned region is held at `vision_path` — the same file the framework derives its
control from — so the request names the source once and it serves both.

Mask reduction to latent resolution takes maxima on the generated side — frame 0 alone,
then groups of `temporal_compression_factor`, spatially `adaptive_max_pool2d` — and then
complements. The pinned side therefore shrinks rather than bleeding into generated
pixels. A 3.5% pixel mask becomes 10% of latent cells.

## Fixed parameters

The checkpoint is causal-distilled: its schedule is baked in. `num_steps` 4,
`guidance` 1, `control_guidance` 1, `shift` 5.0, `sigma_max` 80.0, resolution tier 480,
`num_video_frames_per_chunk` 81, `num_conditional_frames` 0. Do not tune any of these to
influence appearance — steer with the prompt and the control hint.

`roi_pin_strength` defaults to 1.0. A sweep at 1.0 / 0.9 / 0.7 / 0.5 on a 90%-pinned
scene showed no inflection: the metric degraded monotonically as w fell, because the
pinned region's correct answer already agrees with the model's prior, so lowering w only
permits drift. Lower it only when an ablation asks for it.

## Where the control signal comes from

The framework derives edge (and blur) from `vision_path` itself; seg and depth it cannot,
so those still arrive as a precomputed `control_path`. This skill uses the framework's own
derivation, so the hint is produced the way the model was trained to receive it.

Precomputing the same map is **not** an equivalent substitute. Measured on three scenes,
same seed and same mask, the same `medium` (100/200) thresholds on both sides, the only
difference being whether Canny ran in a script beforehand or inside the framework:

| scene | changed, whole frame | changed, generated region | max diff | PSNR |
|---|---|---|---|---|
| wrist camera, tub | 45.7% | 99.7% | 177 | 26.99 dB |
| foreground replace | 24.4% | 98.2% | 151 | 30.27 dB |
| background replace | 57.3% | 99.5% | 146 | 25.33 dB |

The noise floor is 0.08% of pixels at max diff 5 — two runs of identical code on the same
seed. These sit three orders of magnitude outside it.

Thresholds are not the cause: both sides used `medium`. Geometry is not either — the
resolved target is identical both ways. What differs is **when** Canny runs. A script
detects edges at its own working size and the framework then scales that bitmap up to the
model grid; deriving it inside the framework detects edges after the scale, on sharper
pixels. A precomputed map also survives an h264 round trip that an in-memory one never
takes.

So results tuned against a precomputed map do not carry over unchanged. Every edge-related
figure below was measured with precomputed maps; the directions hold, the exact numbers
were not re-measured against the framework-derived hint.

## Why the edge map covers the whole frame

Measured on the drone scene, same prompt and seed, the only difference being whether the
Canny map was zeroed outside the mask:

| edge control | MAE outside mask | MAE inside mask |
|---|---|---|
| zeroed outside the mask | 4.69 | 60.81 |
| full frame | 4.48 | 50.24 |

Both halves improved. The background edges give the model a global geometric reference,
so the replaced objects land in the right place and perspective; with only the object
outlines the model knows a car is there but not where it sits in the scene.

Zeroing also creates a false edge along the mask boundary, which the full-frame map
does not have. The framework's own derivation is full-frame, so this holds either way —
it is a reason not to go back to masking the hint, not a reason to precompute one.

## Why not one composite control channel

Packing "edge inside the mask, source RGB outside" into a single channel fails: 96.5% of
that channel is photographic RGB, so the model reads the channel as an appearance map and
renders the black edge patches as dark objects. Keep the hint a hint and the reference a
reference.

## Environments

Three, and they do not overlap.

| Step | Interpreter | Needs |
|---|---|---|
| prep / mask / request / verify | any Python 3.11+ | numpy, pillow, imageio + imageio-ffmpeg |
| SAM3 mask | a SAM3 environment | transformers ≥ 5.16 with `Sam3VideoModel`, torch, PIL, opencv |
| transfer | the [Framework environment](../../shared/setup.md) | ffmpeg on `PATH` |

Keep the SAM3 environment separate: the transformers version it needs is not the one the
framework pins. The handoff between them is PNG frames on disk, not a shared decoder —
`prep_source.py` writes PNG frames and `extract_mask.py` writes PNG masks, so the SAM3
environment never has to decode or encode video. `build_mask.py` encodes them.

Model and session dtype must match in `extract_mask.py`. Loading float32 weights into a
bfloat16 session makes the memory encoder's conv raise
`Input type (c10::BFloat16) and bias type (float) should be the same`.

## What you have to point at

Nothing in this skill carries a site path. Supply these:

| | How |
|---|---|
| Sim-Transfer snapshot | `SIM_TRANSFER_CHECKPOINT` |
| Cosmos3-Nano `text_tokenizer` | `SIM_TRANSFER_TEXT_TOKENIZER` |
| `Wan2.2_VAE.pth` | `SIM_TRANSFER_VAE` |
| framework checkout | `FRAMEWORK_DIR`, or let `run_transfer.sh` resolve it from its own location |
| SAM3 weights | `SAM3_MODEL`, or `--model` on `extract_mask.py` |

The first four come from [Framework setup](../../shared/setup.md) and the
[matched checkpoint step](../../inference/SKILL.md#1-select-the-matched-inference-checkpoint);
this task adds only the last one.

Two constraints on where an experiment lives:

- **Storage the whole run can see.** On a cluster, node-local scratch such as `/tmp` is
  invisible to a job that lands on another node. Keep experiments on shared storage.
- **Reachable from the transfer step.** If the transfer runs somewhere with a different
  view of the filesystem — a container, another host — the request must carry paths as
  that side sees them. `build_request.py --repo-root LOCAL --container-root REMOTE`
  rewrites them, and refuses to write a request for an experiment outside `--repo-root`
  rather than emit a path that fails at runtime. When both sides see the same paths,
  leave both flags alone.

## Where the ROI pin code lives

`roi_pin_mask_path` and `roi_pin_strength` are implemented in three files of this
repository:

- `cosmos_framework/inference/args.py` — the two fields on `TransferDataArgs` and
  `TransferDataOverrides`, plus a before-validator that coerces a null strength to 1.0
- `cosmos_framework/inference/transfer.py` — reads the mask and reference, encodes the
  reference through the ordinary VAE path, attaches `_roi_pin_mask` / `_roi_pin_strength`
- `cosmos_framework/model/generator/omni_mot_causal_model.py` —
  `build_latent_roi_pin_mask` plus the velocity blend in `_run_distilled_ar_sampler`

Check they are present before running:

```bash
grep -c roi_pin cosmos_framework/inference/args.py
```

A run on a tree without the patch fails with a pydantic validation error naming the
unknown `roi_pin_*` fields. That is the good case. The bad case is a tree where the
fields parse but the sampler does not apply them — which is why a paired `--no-pin`
control run, not a log line, is the discriminator.

That log line is itself easy to lose: `omni_mot_causal_model.py` imports `logger` from
loguru directly, and the framework removes loguru's handlers at startup, so anything
logged through the module-level `log` there is discarded. `build_latent_roi_pin_mask`
imports `cosmos_framework.utils.log` locally for exactly this reason.

## Non-distilled samplers

Pinning is implemented in the distilled autoregressive sampler only. Requesting it with a
non-distilled sampler raises rather than silently ignoring the fields.

## Measured runtimes

| Step | Hardware | Time |
|---|---|---|
| SAM3 mask, 80 frames at 1088² | 1× H20 | ~1 min |
| 4-step transfer, 80 frames | 1× H20 | 3–8 min |
| 50-step transfer, 80 frames | 2× H20 | 2.5–3.3 h |

The 50-step path is not an upgrade for this job: with a blur channel holding the
background it measured 10.21 MAE outside the mask against 4.48 for 4-step with pinning.
