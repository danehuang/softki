#!/bin/bash

DEBUG=true
DATA_DIR=data/uci_datasets/uci_datasets
DATASETS=(pol elevators bike kin40k protein keggdirected slice keggundirected 3droad song buzz houseelectric)
EPOCHS=50

if $DEBUG; then
    EPOCHS=1
fi

for dataset in pol elevators bike kin40k protein keggdirected slice keggundirected 3droad song buzz houseelectric
do
    python gp/train.py --data_dir $DATA_DIR --dataset.name $dataset --training.epochs $EPOCHS
done
