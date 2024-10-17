#!/bin/bash

DEBUG=false

DATA_DIR=data/uci_datasets/uci_datasets
# DATASETS=(pol elevators bike kin40k protein keggdirected slice keggundirected 3droad song buzz houseelectric)
DATASETS=(bike kin40k protein keggdirected slice keggundirected 3droad song buzz houseelectric)
# DATASETS=(elevators)
THRESHOLDS=(0.001 0.0001 0.00001 0.000001)

GROUP=benchmark-threshold-tune
EPOCHS=50
DEVICE="cuda:1"
# SEEDS=(6535 8830 92357)
SEEDS=(6535)

if $DEBUG; then
    EPOCHS=1
fi

for seed in "${SEEDS[@]}"
    do
    for dataset in "${DATASETS[@]}"
    do
        for threshold in "${THRESHOLDS[@]}"
        do
            python train.py \
                --model.name soft-gp \
                --model.num_inducing 512 \
                --model.device $DEVICE \
                --model.use_qr \
                --model.use_scale \
                --model.T 1 \
                --model.threshold $threshold \
                --model.use_threshold \
                --data_dir $DATA_DIR \
                --dataset.name $dataset \
                --training.seed $seed \
                --training.epochs $EPOCHS \
                --training.learning_rate 0.01 \
                --dataset.train_frac 0.9 \
                --dataset.val_frac 0 \
                --wandb.group $GROUP \
                --wandb.watch
        done
    done
done
