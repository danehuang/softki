#!/bin/bash

DEBUG=false

DATA_DIR=data/
DATASETS=(Ac-Ala3-NHMe AT-AT-CG-CG AT-AT stachyose DHA buckyball-catcher double-walled-nanotube)

PROJECT=softki2
GROUP=md22-ard
EPOCHS=50
DEVICE="cuda:1"
SEEDS=(6535 8830 92357)

if $DEBUG; then
    EPOCHS=1
fi

pushd ..
    for dataset in "${DATASETS[@]}"
    do
        for seed in "${SEEDS[@]}"
        do
            python run.py \
                --model.name softki \
                --model.num_inducing 512 \
                --model.device $DEVICE \
                --model.use_qr \
                --model.use_ard \
                --model.use_scale \
                --model.T 1 \
                --data_dir $DATA_DIR \
                --dataset.name $dataset \
                --training.seed $seed \
                --training.epochs $EPOCHS \
                --training.learning_rate 0.01 \
                --dataset.train_frac 0.9 \
                --dataset.val_frac 0 \
                --wandb.project $PROJECT \
                --wandb.group $GROUP \
                --wandb.watch

            python run.py \
                --model.name svgp \
                --model.num_inducing 1024 \
                --model.device $DEVICE \
                --model.learn_noise \
                --model.use_ard \
                --model.use_scale \
                --data_dir $DATA_DIR \
                --dataset.name $dataset \
                --training.seed $seed \
                --training.epochs $EPOCHS \
                --training.learning_rate 0.01 \
                --dataset.train_frac 0.9 \
                --dataset.val_frac 0 \
                --wandb.project $PROJECT \
                --wandb.group $GROUP \
                --wandb.watch

            python run.py \
                --model.name sgpr \
                --model.num_inducing 512 \
                --model.device $DEVICE \
                --model.learn_noise \
                --model.use_ard \
                --model.use_scale \
                --data_dir $DATA_DIR \
                --dataset.name $dataset \
                --training.seed $seed \
                --training.epochs $EPOCHS \
                --training.learning_rate .1 \
                --dataset.train_frac 0.9 \
                --dataset.val_frac 0 \
                --wandb.project $PROJECT \
                --wandb.group $GROUP \
                --wandb.watch
        done
    done
popd