#!/bin/bash

DEBUG=false

DATA_DIR=data/
DATASETS=(Ac-Ala3-NHMe AT-AT-CG-CG AT-AT stachyose DHA)

GROUP=md22
EPOCHS=50
NUM_INDUCING=1024
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
            --model.name soft-gp \
            --model.num_inducing $NUM_INDUCING \
            --model.device $DEVICE \
            --model.use_qr \
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

        python train.py \
            --model.name svi-gp \
            --model.num_inducing $NUM_INDUCING \
            --model.device $DEVICE \
            --model.learn_noise \
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

        python train.py \
            --model.name sv-gp \
            --model.num_inducing $NUM_INDUCING \
            --model.device $DEVICE \
            --model.learn_noise \
            --model.use_scale \
            --data_dir $DATA_DIR \
            --dataset.name $dataset \
            --training.seed $seed \
            --training.epochs $EPOCHS \
            --training.learning_rate .1 \
            --dataset.train_frac 0.9 \
            --dataset.val_frac 0 \
            --wandb.group $GROUP \
            --wandb.watch
    done
done