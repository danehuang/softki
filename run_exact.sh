#!/bin/bash

DEBUG=false

DATA_DIR=data/uci_datasets/uci_datasets
DATASETS=(pol elevators bike kin40k protein keggdirected slice keggundirected 3droad song buzz houseelectric)

GROUP=benchmark7
EPOCHS=50
NUM_INDUCING=512
DEVICE="cuda:1"
SEEDS=(6535 8830 92357)

if $DEBUG; then
    EPOCHS=1
fi

for dataset in "${DATASETS[@]}"
do
    for seed in "${SEEDS[@]}"
    do
        python train.py \
            --model.name exact \
            --model.device $DEVICE \
            --model.use_scale \
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