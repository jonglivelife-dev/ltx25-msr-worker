# ComfyUI as a serverless worker, with the one custom node this workflow needs.
#
# The weights are NOT baked in — 72 GB would make an unusable image and a cold
# start measured in minutes. They live on a network volume mounted at
# /runpod-volume, which ComfyUI is pointed at below. Only the code is here.
# Pinned: this repo publishes no :latest, and an unpinned base is a silent
# rebuild waiting to happen. 5.10.0-base carries no bundled model.
FROM runpod/worker-comfyui:5.10.0-base

# ComfyUI-LTX2.5-MSR — multi-reference conditioning. Its pyproject declares
# dependencies = [], so there is nothing to pip install after it.
RUN cd /comfyui/custom_nodes \
 && git clone --depth 1 https://github.com/liconstudio/ComfyUI-LTX2.5-MSR.git

# Read every model class from the volume. Written at build time so a cold start
# does not depend on a file that may or may not have been copied in.
RUN printf '%s\n' \
  'runpod_volume:' \
  '  base_path: /runpod-volume/' \
  '  checkpoints: models/checkpoints' \
  '  diffusion_models: models/diffusion_models' \
  '  text_encoders: models/text_encoders' \
  '  clip: models/text_encoders' \
  '  vae: models/vae' \
  '  loras: models/loras' \
  '  latent_upscale_models: models/latent_upscale_models' \
  '  upscale_models: models/upscale_models' \
  > /comfyui/extra_model_paths.yaml

# Fill the volume on first start rather than baking 72 GB into the image.
COPY fetch-models.sh /usr/local/bin/fetch-models.sh
RUN chmod +x /usr/local/bin/fetch-models.sh
ENTRYPOINT ["/usr/local/bin/fetch-models.sh"]
CMD ["/start.sh"]
