#!/bin/bash
set -uo pipefail
set +e

FAILURE=false

echo "running full loop test with CNN on fake data"
python training/run_experiment.py --data_class=FakeImageData --model_class=CNN --conv_dim=2 --fc_dim=2 --loss=cross_entropy --num_workers=4 --max_epochs=1 || FAILURE=true

echo "running fast_dev_run test of real model class on real data"
python training/run_experiment.py --data_class=IAMParagraphs --model_class=ResnetTransformer --loss=transformer \
  --tf_dim 4 --tf_fc_dim 2 --tf_layers 2 --tf_nhead 2 --batch_size 2 --lr 0.0001 \
  --fast_dev_run --num_sanity_val_steps 0 \
  --num_workers 1 || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Test for run_experiment.py failed"
  exit 1
fi
echo "Tests for run_experiment.py passed"
exit 0
