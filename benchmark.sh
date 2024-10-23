#!/bin/bash

# Run benchmark scripts
pushd scripts
    ./run_uci.sh
    ./run_md22.sh
    ./run_noise.sh
popd
