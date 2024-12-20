#!/bin/bash

DEBUG=true

DATA_DIR=data/uci_datasets/uci_datasets
DATASETS=(pol elevators bike kin40k protein keggdirected slice keggundirected 3droad song buzz houseelectric)

PROJECT=softki2
GROUP=nonoise
EPOCHS=50
DEVICE="cuda:0"
NUM_WORKERS=0
SEEDS=(6535 8830 92357)

if $DEBUG; then
    EPOCHS=1
    SEEDS=(6535)
fi

pushd ..
    for seed in "${SEEDS[@]}"
        do
        for dataset in "${DATASETS[@]}"
        do
            python run.py \
                --model.name svgp \
                --model.num_inducing 1024 \
                --model.device $DEVICE \
                --model.use_scale \
                --data_dir $DATA_DIR \
                --dataset.name $dataset \
                --training.seed $seed \
                --training.epochs $EPOCHS \
                --training.learning_rate 0.01 \
                --dataset.train_frac 0.9 \
                --dataset.val_frac 0 \
                --dataset.num_workers $NUM_WORKERS \
                --wandb.project $PROJECT \
                --wandb.group $GROUP \
                --wandb.watch

            python run.py \
                --model.name sgpr \
                --model.num_inducing 512 \
                --model.device $DEVICE \
                --model.use_scale \
                --data_dir $DATA_DIR \
                --dataset.name $dataset \
                --training.seed $seed \
                --training.epochs $EPOCHS \
                --training.learning_rate .1 \
                --dataset.train_frac 0.9 \
                --dataset.val_frac 0 \
                --dataset.num_workers $NUM_WORKERS \
                --wandb.project $PROJECT \
                --wandb.group $GROUP \
                --wandb.watch
        done
    done
popd