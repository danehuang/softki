#!/bin/bash

DEBUG=false

DATA_DIR=data/uci_datasets/uci_datasets
DATASETS=(pol elevators bike kin40k protein keggdirected slice keggundirected 3droad song buzz houseelectric)

GROUP=inducing
EPOCHS=50
NUM_INDUCING=(64 128 256 512 1024 1536 2048 4096)
BATCH_SIZE=(1024)
DEVICE="cuda:0"
SEEDS=(6535 8830 92357)

if $DEBUG; then
    EPOCHS=1
fi

for dataset in "${DATASETS[@]}"
do
    
    for num_inducing in "${NUM_INDUCING[@]}"
    do
        for seed in "${SEEDS[@]}"
        do
            for batch_size in "${BATCH_SIZE[@]}"
            do
                python train.py \
                    --model.name soft-gp \
                    --model.num_inducing $num_inducing \
                    --model.device $DEVICE \
                    --model.use_qr \
                    --model.use_scale \
                    --data_dir $DATA_DIR \
                    --dataset.name $dataset \
                    --training.seed $seed \
                    --training.epochs $EPOCHS \
                    --training.batch_size $batch_size \
                    --training.learning_rate 0.01 \
                    --dataset.train_frac 0.9 \
                    --dataset.val_frac 0 \
                    --wandb.group $GROUP \
                    --wandb.watch
            done
        done
    done
done