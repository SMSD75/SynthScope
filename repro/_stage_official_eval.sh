#!/bin/bash
S=/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared/pgc_official_eval
cp -r /tmp/pgc_official $S/repo
cp /hddstore/ssalehi/SynthScope/repro/official_ckpts/pgc/PGC_train_progan_ckpt.pth $S/
SNAP=$(ls -d /hddstore/ssalehi/SynthScope/.cache/huggingface/hub/models--facebook--dinov2-large/snapshots/*/ | head -1)
mkdir -p $S/pretrained_dino/dinov2-large
cp -rL "$SNAP"/* $S/pretrained_dino/dinov2-large/
mkdir -p $S/test
for g in deepfake san seeingdark; do cp -r /hddstore/ssalehi/SynthScope/CNNDetection/test/$g $S/test/; done
touch $S/READY
echo "staged: $(du -sh $S | cut -f1)"
