#!/usr/bin/env bash
# Cosmos3-Nano-Sim-Transfer, the causal-distilled 4-step checkpoint, with ROI latent
# pinning. Runs every request JSON in REQUESTS, one per visible GPU, and skips any that
# already has output, so an interrupted wave can simply be re-run.
#
# The flags below come from the checkpoint's bundled recipe and are not free parameters;
# appearance is steered through the prompt and the control hint only.
#
# Required:
#   REQUESTS                       directory of request JSONs
#   OUT                            output directory
#   SIM_TRANSFER_CHECKPOINT        release directory
#   SIM_TRANSFER_TEXT_TOKENIZER    Cosmos3-Nano text_tokenizer directory
#   SIM_TRANSFER_VAE               Wan2.2_VAE.pth
#
# The last three are the ones ../../inference/SKILL.md exports; follow its
# "Select the matched inference checkpoint" step and they are already set.
#
# Optional:
#   FRAMEWORK_DIR  framework checkout (default: four levels above this skill)
#   HISTORY        KV-cache inference size (default 0)
#   SINK           attention sink size (default 0)
#
# This script takes no scheduler directives. Run it directly, or wrap it in whatever your
# site uses -- a batch job, a container, a plain shell. It needs the framework's own
# `.venv`, so run it where that venv works (for the NGC PyTorch image, inside it).

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/ -> real2real-replace -> cosmos3-nano-sim-transfer -> skills -> .agents -> root
F="${FRAMEWORK_DIR:-$(cd "$HERE/../../../../.." && pwd)}"
SKILL="$F/.agents/skills/cosmos3-nano-sim-transfer"
CKPT="${SIM_TRANSFER_CHECKPOINT:?set SIM_TRANSFER_CHECKPOINT; see ../../inference/SKILL.md}"
REQUESTS="${REQUESTS:?set REQUESTS}"
OUT="${OUT:?set OUT}"
# The release ships its own text tokenizer and VAE, but the recipe pins the Cosmos3-Nano
# tokenizer and the Wan 2.2 VAE; keep that pairing.
TOKENIZER="${SIM_TRANSFER_TEXT_TOKENIZER:?set SIM_TRANSFER_TEXT_TOKENIZER}"
VAE="${SIM_TRANSFER_VAE:?set SIM_TRANSFER_VAE}"
HISTORY="${HISTORY:-0}"; SINK="${SINK:-0}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset LD_LIBRARY_PATH PYTHONPATH COSMOS_DISTILLATION_IMPL

# torchcodec dlopens libavutil at import; without ffmpeg every sample dies at startup.
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found: torchcodec dlopens libavutil at import and every sample will die at startup" >&2
  exit 2
fi
cd "$F"

# Same list as SIM_TRANSFER_RESOURCE_OVERRIDES in ../../inference/SKILL.md.
OVERRIDES=(
  model.config.tokenizer.bucket_name=
  model.config.tokenizer.object_store_credential_path_pretrained=
  "model.config.tokenizer.vae_path=$VAE"
  model.config.vlm_config.tokenizer.config_variant=hf
  "model.config.vlm_config.tokenizer.pretrained_model_name=$TOKENIZER"
  model.config.vlm_config.pretrained_weights.enabled=False
  model.config.teacher_forcing_frames_per_chunk=1
  model.config.kv_cache_dtype=null
)

NGPU="$(nvidia-smi -L | wc -l)"
# Resumable: a request that already has output is skipped, so a wave can be re-submitted
# after a timeout or a lost node without redoing finished work.
samples=()
for spec in "$REQUESTS"/*.json; do
  tag="$(basename "$spec" .json)"
  [[ -f "$OUT/$tag/$tag/vision.mp4" ]] && continue
  samples+=("$spec")
done
(( ${#samples[@]} == 0 )) && { echo "nothing left to do"; exit 0; }
mkdir -p "$OUT/logs" "$OUT/.claims"
echo "launching ${#samples[@]} requests over $NGPU GPUs"

run_one() {
  local spec="$1" gpu="$2" tag
  tag="$(basename "$spec" .json)"
  # mkdir is atomic, so exactly one worker claims a request even across nodes.
  if ! mkdir "$OUT/.claims/$tag" 2>/dev/null; then
    echo "[gpu $gpu] skip $tag (claimed elsewhere)"
    return 0
  fi
  export CUDA_VISIBLE_DEVICES="$gpu"
  export XDG_CACHE_HOME="$F/.cache/xdg/$tag" TORCHINDUCTOR_CACHE_DIR="$F/.cache/inductor/$tag"
  mkdir -p "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR"
  echo "[gpu $gpu] start $tag"
  local rc=0
  set +e
  .venv/bin/python -m cosmos_framework.scripts.inference \
    -i "$spec" -o "$OUT/$tag" \
    --checkpoint-path "$CKPT" --config-file "$CKPT/config.json" --credential-path "" \
    --experiment-overrides "${OVERRIDES[@]}" \
    --defaults-file "$SKILL/inference/assets/sample_args.json" \
    --autoregressive=True \
    --cp-size 1 --cfgp-size 1 --dp-shard-size 1 --dp-replicate-size 1 \
    --num-outputs 1 --max-num-seqs 1 \
    --num-steps 4 --guidance 1 --control-guidance 1 \
    --kv-cache-inference-size "$HISTORY" --attention-sink-size "$SINK" \
    --no-use-torch-compile --no-use-cuda-graphs --no-condition-stream-encode \
    --no-diffusion-cache --prompt-upsampling=False --no-guardrails \
    >"$OUT/logs/$tag.log" 2>&1
  rc=$?
  set -e
  (( rc != 0 )) && rmdir "$OUT/.claims/$tag" 2>/dev/null
  echo "[gpu $gpu] done $tag rc=$rc"
  return $rc
}

declare -A slot_pid=()
status=0; next=0
while (( next < ${#samples[@]} )) || (( ${#slot_pid[@]} > 0 )); do
  while (( next < ${#samples[@]} )) && (( ${#slot_pid[@]} < NGPU )); do
    for (( g=0; g<NGPU; g++ )); do
      if [[ -z "${slot_pid[$g]:-}" ]]; then
        run_one "${samples[$next]}" "$g" & slot_pid[$g]=$!
        next=$(( next + 1 )); break
      fi
    done
  done
  wait -n || status=1
  for g in "${!slot_pid[@]}"; do
    if ! kill -0 "${slot_pid[$g]}" 2>/dev/null; then
      wait "${slot_pid[$g]}" 2>/dev/null || status=1
      unset 'slot_pid['"$g"']'
    fi
  done
done
exit "$status"
