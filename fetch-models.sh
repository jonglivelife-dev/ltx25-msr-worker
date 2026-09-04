#!/usr/bin/env bash
# Fill the network volume on first start, then never again.
#
# The weights are 72 GB, which is far too much for a Docker image and far too
# much to push up from a laptop. Downloading them here happens once, inside the
# datacentre, at datacentre speed — and every later cold start finds the files
# already present and skips straight past.
set -uo pipefail
V=/runpod-volume
[ -d "$V" ] || { echo "no network volume mounted at $V — nothing to fill"; exec "$@"; }

L=https://huggingface.co/Lightricks/LTX-2.5/resolve/main
M=https://huggingface.co/LiconStudio/LTX-2.5-Multiple-Subject-Reference/resolve/main

mkdir -p "$V"/models/{diffusion_models,text_encoders,vae,latent_upscale_models,loras/ltx2.5}

fetch () {   # url  dest  expected-bytes
  local url=$1 dest=$2 want=$3 path="$V/$2"
  if [ -f "$path" ]; then
    local have; have=$(stat -c%s "$path" 2>/dev/null || echo 0)
    # A part-finished download from a killed worker must not be trusted.
    if [ "$have" -ge "$want" ]; then echo "have  $(basename "$dest")"; return; fi
    echo "resume $(basename "$dest")  ($have/$want)"
  else
    echo "fetch $(basename "$dest")"
  fi
  curl -fL --retry 5 --retry-delay 5 -C - -o "$path" "$url" || echo "FAILED $dest"
}

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
exec "$@"
