#!/bin/bash

DEBUG=false

DATA_DIR=data/uci_datasets/uci_datasets
DATASETS=(pol elevators bike kin40k protein keggdirected slice keggundirected 3droad song buzz houseelectric)

PROJECT=softki2
GROUP=benchmark-scale
EPOCHS=50
DEVICE="cuda:1"
NUM_WORKERS=0
SEEDS=(6535 8830 92357)

if $DEBUG; then
    EPOCHS=1
fi

pushd ..
    for seed in "${SEEDS[@]}"
        do
        for dataset in "${DATASETS[@]}"
        do
            python train.py \
                --model.name soft-gp \
                --model.num_inducing 512 \
                --model.device $DEVICE \
                --model.scale_softmax \
                --model.use_qr \
                --model.use_scale \
                --model.T 1 \
                --model.mll_approx exact \
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
        done
    done
popd