#!/usr/bin/env bash
# Fill the network volume on first start, then never again.
#
# 147 GB of weights: LTX-2.5, MiniMax H3 (ref2va) and the VDN hybrid-attention
# checkpoint. Far too much for a Docker image and far too much to push up from a
# laptop, so it is fetched here once, inside the datacentre, and every later
# cold start finds the files present and skips straight past.
#
# A LOCK MATTERS HERE. Several workers mount this same volume, and eight of them
# resuming the same partial download would corrupt it rather than fail loudly.
# The first worker to arrive fills; the rest wait for it, then read.
set -uo pipefail
V=/runpod-volume
[ -d "$V" ] || { echo "no network volume at $V — nothing to fill"; exec "$@"; }

LOCK="$V/.fetching"
DONE="$V/.models-complete"

if [ -f "$DONE" ]; then
  echo "models already present"; exec "$@"
fi

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another worker is filling the volume — waiting"
  for _ in $(seq 1 240); do            # up to 2 hours
    [ -f "$DONE" ] && { echo "fill finished elsewhere"; exec "$@"; }
    sleep 30
  done
  echo "gave up waiting on the other worker"; exec "$@"
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

L=https://huggingface.co/Lightricks/LTX-2.5/resolve/main
C=https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main
D=https://huggingface.co/OpenVDN/vdn-minimax-h3/resolve/main
M=https://huggingface.co/LiconStudio/LTX-2.5-Multiple-Subject-Reference/resolve/main

mkdir -p "$V"/models/{diffusion_models,text_encoders,vae,latent_upscale_models,loras/ltx2.5,vdn}

fetch () {   # url  dest  expected-bytes
  local url=$1 dest=$2 want=$3 path="$V/$2"
  mkdir -p "$(dirname "$path")"
  if [ -f "$path" ]; then
    local have; have=$(stat -c%s "$path" 2>/dev/null || echo 0)
    if [ "$have" -ge "$want" ]; then echo "have   $(basename "$dest")"; return; fi
    echo "resume $(basename "$dest")  ($have/$want)"
  else
    echo "fetch  $(basename "$dest")"
  fi
  curl -fL --retry 5 --retry-delay 5 -C - -o "$path" "$url" || echo "FAILED $dest"
}

echo "=== MiniMax H3 — the model under test ==="
fetch "$C/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors" \
      "models/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors" 34000000000
fetch "$C/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors" \
      "models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors" 27000000000
fetch "$C/vae/minimax_h3_video_vae_fp16.safetensors" \
      "models/vae/minimax_h3_video_vae_fp16.safetensors" 5200000000
fetch "$C/vae/minimax_h3_audio_vae_fp32.safetensors" \
      "models/vae/minimax_h3_audio_vae_fp32.safetensors" 600000000
fetch "$C/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors" \
      "models/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors" 1900000000

echo "=== VDN — hybrid attention, for the fights ==="
for f in adapters/default/adapter_config.json adapters/default/adapter_model.safetensors \
         adapters/turbo/adapter_config.json adapters/turbo/adapter_model.safetensors \
         linear_branch/config.json linear_branch/model.safetensors \
         metadata.json model_spec.json; do
  fetch "$D/stage-dmd-step-250/$f" "models/vdn/stage-dmd-step-250/$f" 1
done

echo "=== LTX-2.5 — the speed comparison ==="
fetch "$L/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
      "models/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" 42020000000
fetch "$L/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
      "models/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" 26260000000
fetch "$L/vae/ltx-2.5-video-vae-bf16.safetensors" \
      "models/vae/ltx-2.5-video-vae-bf16.safetensors" 1470000000
fetch "$L/vae/ltx-2.5-audio-vae-bf16.safetensors" \
      "models/vae/ltx-2.5-audio-vae-bf16.safetensors" 360000000
fetch "$L/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
      "models/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" 1000000000
fetch "$M/LTX-2.5-Licon-MSR-V1.safetensors" \
      "models/loras/ltx2.5/LTX-2.5-Licon-MSR-V1.safetensors" 1300000000

du -sh "$V"/models/* 2>/dev/null
touch "$DONE"
echo "volume filled"
exec "$@"
