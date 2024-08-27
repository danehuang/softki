#!/bin/bash

DEBUG=false

DATA_DIR=data/uci_datasets/uci_datasets
DATASETS=(pol elevators bike kin40k protein keggdirected slice keggundirected 3droad song buzz houseelectric)

GROUP=benchmark
EPOCHS=50
LEARNING_RATE=0.01
NUM_INDUCING=512
DEVICE="cuda:0"

if $DEBUG; then
    EPOCHS=1
fi

for dataset in "${DATASETS[@]}"
do
    python train.py \
        --model.name soft-gp \
        --model.num_inducing $NUM_INDUCING \
        --model.device $DEVICE \
        --data_dir $DATA_DIR \
        --dataset.name $dataset \
        --training.epochs $EPOCHS \
        --training.learning_rate $LEARNING_RATE \
        --wandb.group $GROUP

    python train.py \
        --model.name svi-gp \
        --model.num_inducing $NUM_INDUCING \
        --model.device $DEVICE \
        --data_dir $DATA_DIR \
        --dataset.name $dataset \
        --training.epochs $EPOCHS \
        --training.learning_rate $LEARNING_RATE \
        --wandb.group $GROUP
done
