#!/bin/bash

for arg in "$@"; do
    if [[ $arg -eq 1 ]]; then
        echo "Input is equal to 1. Running prior_s1"
        bash run_train2.sh train_s1.py prior_s1 --v2c_projector --bidir_mixco

    else
        echo "Invalid input. Ignoring argument $arg."
    fi
done