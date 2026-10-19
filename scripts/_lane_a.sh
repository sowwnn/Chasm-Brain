#!/usr/bin/env bash
while pgrep -f "train_hierarchical_v3.py --config configs/train_tben_two_stage_subj5" > /dev/null; do sleep 30; done
exec ./scripts/run_tben_s2_lane.sh 1
