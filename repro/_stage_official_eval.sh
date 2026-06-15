#!/bin/bash
S=/ivi/zfs/s0/original_homes/ssalehi/AGID_shared/pgc_official_eval
cp -r /tmp/pgc_official $S/repo
cp /hddstore/ssalehi/AGID/repro/official_ckpts/pgc/PGC_train_progan_ckpt.pth $S/
SNAP=$(ls -d /hddstore/ssalehi/AGID/.cache/huggingface/hub/models--facebook--dinov2-large/snapshots/*/ | head -1)
mkdir -p $S/pretrained_dino/dinov2-large
cp -rL "$SNAP"/* $S/pretrained_dino/dinov2-large/
mkdir -p $S/test
for g in deepfake san seeingdark; do cp -r /hddstore/ssalehi/AGID/CNNDetection/test/$g $S/test/; done
touch $S/READY
echo "staged: $(du -sh $S | cut -f1)"
