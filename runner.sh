#!/bin/sh

# ONLY FOR TRAINING EXPERIMENTS
for i in $(seq "$1" "$2")
do
  echo "Running experiment_num: $i"
  python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port 9999 ./run_training.py --experiment_number "$i"
done
