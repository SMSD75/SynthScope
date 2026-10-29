#!/bin/bash
# VIB-Net (Zhang et al., CVPR 2025), its training recipe reconstructed in SynthScope (no official training code exists):
#   data      GenImage Stable Diffusion v1.4 train split, a seeded 16k real + 16k fake subset (the paper: "32,000 real
#             and generated images"); validation = a disjoint 1k + 1k of the same train split (the GenImage val alias is
#             its test split). The GAN setting trains on ForenSynths' 20-class ProGAN set (dataset image/cnndetection).
#   train     blur p0.1 (sigma 0-3) + JPEG p0.1 (cv2/PIL, q30-100) -> random 224 crop at native resolution (padded if
#             smaller) -> flip -> CLIP normalization; validation: blur sigma 1.5 / JPEG q65 at the same rates -> center
#             crop 224. VIB-Net's released option values on UniversalFakeDetect's augmentation, without its resize to
#             256: the official test is a 224 center crop of the image at its own size (main.py), and training at that
#             scale is what reproduces the paper (with the resize, GenImage 512 px images are seen at half scale in
#             training: 15-generator AP 82.98 instead of 94.08)
#   model     frozen fp32 CLIP ViT-L/14 + the VIB head exactly as the official model (conf/model/image/vib_net_official)
#   optimizer Adam lr 2e-4, no weight decay, batch 256, plateau schedule on validation accuracy (conf/trainer/image/vib_net)
# Evaluate:  python benchmark/bench_eval.py --model vib_net_official --ckpt <save_path>/model_best.pth --benchmark pergen --data <GenImage_full/test>
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/nvmestore/ssalehi/.triton}
export HF_HOME=${HF_HOME:-/nvmestore/ssalehi/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/nvmestore/ssalehi/.cache/torch}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=1
DATA=${DATA:-/nvmestore/ssalehi/SynthScope/GenImage}
SAVE=${SAVE:-/nvmestore/ssalehi/SynthScope/models/vib_net}

# Plain single-process run: the repo's accelerate_config.yaml (DeepSpeed) would add gradient clipping.
/home/ssalehi/miniconda3/bin/conda run --no-capture-output -n SynthScope python main.py \
    trainer@TrainerConfig=image/vib_net \
    model@ModelConfig=image/vib_net_official \
    dataset@DatasetConfig=image/genimage \
    DatasetConfig.root_dir=$DATA \
    +DatasetConfig.train_per_class=16000 \
    +DatasetConfig.val_per_class=1000 \
    DatasetConfig.prefetch=false \
    DatasetConfig.train_batch_size=256 \
    DatasetConfig.val_batch_size=256 \
    "DatasetConfig.train_transform_list=[blur_jpeg_cnndet,pad_random_crop_224,random_horizontal_flip,to_dtype,normalize_clip]" \
    "DatasetConfig.eval_transform_list=[blur_jpeg_cnndet_val,center_crop_224,to_dtype,normalize_clip]" \
    TrainerConfig.save_path=$SAVE \
    "metrics=[real_accuracy,fake_accuracy]" \
    logger.name=vib_net \
    logger.mode=offline \
    "logger.tags=[SynthScope,VIBNet,genimage_sd14]" \
    "$@"
