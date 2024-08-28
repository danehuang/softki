#!/bin/bash

DEBUG=false

DATA_DIR=data/uci_datasets/uci_datasets
# DATASETS=(pol elevators bike kin40k protein keggdirected slice keggundirected 3droad song buzz houseelectric)
# DATASETS=(slice 3droad song buzz houseelectric)
DATASETS=(3droad buzz)

EPOCHS=50
LEARNING_RATE=0.01
NUM_INDUCING=512
DEVICE="cuda:0"
GROUP=qr

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
        --wandb.watch \
        --wandb.group $GROUP

    # python train.py \
    #     --model.name soft-gp \
    #     --model.num_inducing $NUM_INDUCING \
    #     --model.device $DEVICE \
    #     --data_dir $DATA_DIR \
    #     --dataset.name $dataset \
    #     --training.epochs $EPOCHS \
    #     --training.learning_rate 0.01 \
    #     --wandb.watch \
    #     --wandb.group $GROUP
done
