# SynthScope web page as a Hugging Face Docker Space (CPU hardware, e.g. "CPU Upgrade": 8 vCPU, 32 GB).
# Build context = the repository root (push_space.py copies this file to the root of the Space repo).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 \
    SYNTHSCOPE_CKPT_DIR=/home/user/ckpts SYNTHSCOPE_CACHE_DIR=/home/user/cache HF_HOME=/home/user/cache/hf \
    SYNTHSCOPE_MAX_UPLOAD_MB=20 SYNTHSCOPE_MAX_RESIDENT_MODELS=2 XFORMERS_DISABLED=1 SYNTHSCOPE_DEMO=1 \
    DINO_V3_REPO_DIR=/home/user/dinov3 \
    DINO_V3_WEIGHTS=/home/user/ckpts/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    DINO_V3_WEIGHTS_DIR=/home/user/ckpts/dinov3 SYNTHSCOPE_MFM_CKPT=/home/user/ckpts/mfm/mfm_pretrain_vit_base.pth
WORKDIR /home/user/app

# DINOv3 backbone code for STALL and the DINOv3 probe (the weights are fetched at start), pinned to the commit SynthScope is verified with
RUN git clone -q https://github.com/facebookresearch/dinov3 /home/user/dinov3 && git -C /home/user/dinov3 checkout -q 31703e4

# CPU-only PyTorch first (much smaller than the CUDA build), then SynthScope with the web extras
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user . .
RUN pip install --no-cache-dir --user -e ".[web]" "git+https://github.com/openai/CLIP.git"

EXPOSE 7860
CMD ["bash", "deploy/hf-space/start.sh"]
