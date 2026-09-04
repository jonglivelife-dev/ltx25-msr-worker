# ComfyUI as a serverless worker for MiniMax H3 and LTX-2.5.
#
# The weights are NOT baked in — 147 GB would make an unusable image and a cold
# start measured in tens of minutes. They live on a network volume mounted at
# /runpod-volume, fetched once on first start. Only code is here.
#
# Custom nodes MUST be here rather than on the volume: ComfyUI scans
# custom_nodes at startup and cannot load them from a mounted path.
FROM runpod/worker-comfyui:5.10.0-base

# VDN-H3 — hybrid attention. Linear cost over clip length, and it keeps fast
# motion clean where a turbo distillation goes to mush.
# KJNodes — MiniMaxChunkFeedForward, ModelPatchTorchSettings, SageAttention,
#   MiniMaxLowVRAMAttention, ImageResizeKJv2, and the Set/Get wiring the
#   reference workflows are built on.
# VideoHelperSuite — VHS_VideoCombine, which is what actually writes the mp4.
# rgthree — Label nodes the shared workflows carry.
RUN cd /comfyui/custom_nodes \
 && git clone --depth 1 https://github.com/Saganaki22/ComfyUI-VDN-H3.git \
 && git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git \
 && git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
 && git clone --depth 1 https://github.com/rgthree/rgthree-comfy.git \
 && git clone --depth 1 https://github.com/liconstudio/ComfyUI-LTX2.5-MSR.git \
 && for d in */requirements.txt; do pip install --no-cache-dir -r "$d" || true; done

# Read every model class from the volume. Written at build time so a cold start
# never depends on a file that may or may not have been copied in.
RUN printf '%s\n' \
  'runpod_volume:' \
  '  base_path: /runpod-volume/' \
  '  checkpoints: models/checkpoints' \
  '  diffusion_models: models/diffusion_models' \
  '  unet: models/diffusion_models' \
  '  text_encoders: models/text_encoders' \
  '  clip: models/text_encoders' \
  '  vae: models/vae' \
  '  loras: models/loras' \
  '  vdn: models/vdn' \
  '  latent_upscale_models: models/latent_upscale_models' \
  '  upscale_models: models/upscale_models' \
  > /comfyui/extra_model_paths.yaml

# Fill the volume on first start rather than baking 147 GB into the image.
COPY fetch-models.sh /usr/local/bin/fetch-models.sh
RUN chmod +x /usr/local/bin/fetch-models.sh
ENTRYPOINT ["/usr/local/bin/fetch-models.sh"]
CMD ["/start.sh"]
