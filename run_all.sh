#!/bin/bash

# Get data
pushd data
    python get_uci.py
    python get_md22.p
popd

# Run benchmark scripts
pushd scripts
    ./run_uci.sh
    ./run_md22.sh
    ./run_noise.sh
popd
