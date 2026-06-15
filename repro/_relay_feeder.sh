#!/bin/bash
RELAY=/ivi/zfs/s0/original_homes/ssalehi/AGID_shared/relay
until [ -f $RELAY/USE_RELAY ]; do [ -f $RELAY/SSH_OK ] && exit 0; sleep 10; done
i=0
tar -C /hddstore/ssalehi/AGID/CNNDetection -cf - train test | while :; do
  while [ $(ls $RELAY/*.chunk 2>/dev/null | wc -l) -ge 2 ]; do sleep 5; done
  p=$RELAY/chunk_$(printf %05d $i).part
  dd of="$p" bs=4M count=3072 iflag=fullblock 2>/dev/null
  sz=$(stat -c %s "$p" 2>/dev/null || echo 0)
  if [ "$sz" -eq 0 ]; then rm -f "$p"; touch $RELAY/DONE; break; fi
  mv "$p" "${p%.part}.chunk"; i=$((i+1))
  [ "$sz" -lt 12884901888 ] && { touch $RELAY/DONE; break; }
done
