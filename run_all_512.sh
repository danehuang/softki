#!/bin/bash

DEBUG=false

DATA_DIR=data/uci_datasets/uci_datasets
DATASETS=(pol elevators bike kin40k protein keggdirected slice keggundirected 3droad song buzz houseelectric)

GROUP=benchmark3
EPOCHS=50
NUM_INDUCING=512
DEVICE="cuda:1"
EIGHT_NINTH=0.888888888

if $DEBUG; then
    EPOCHS=1
fi

for dataset in "${DATASETS[@]}"
do
    python train.py \
        --model.name soft-gp \
        --model.num_inducing $NUM_INDUCING \
        --model.device $DEVICE \
        --model.use_qr \
        --data_dir $DATA_DIR \
        --dataset.name $dataset \
        --training.epochs $EPOCHS \
        --training.learning_rate 0.01 \
        --dataset.train_frac $EIGHT_NINTH \
        --dataset.val_frac 0 \
        --wandb.group $GROUP \
        --wandb.watch

    python train.py \
        --model.name svi-gp \
        --model.num_inducing $NUM_INDUCING \
        --model.device $DEVICE \
        --data_dir $DATA_DIR \
        --dataset.name $dataset \
        --training.epochs $EPOCHS \
        --training.learning_rate 0.01 \
        --dataset.train_frac $EIGHT_NINTH \
        --dataset.val_frac 0 \
        --wandb.group $GROUP \
        --wandb.watch

    python train.py \
        --model.name sv-gp \
        --model.num_inducing $NUM_INDUCING \
        --model.device $DEVICE \
        --data_dir $DATA_DIR \
        --dataset.name $dataset \
        --training.epochs $EPOCHS \
        --training.learning_rate .1 \
        --wandb.group $GROUP \
        --dataset.train_frac $EIGHT_NINTH \
        --dataset.val_frac 0 \
        --wandb.watch
done
