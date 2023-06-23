#!/bin/bash

for arg in "$@"; do
    if [[ $arg -eq 1 ]]; then
        echo "Input is equal to 1. Running s3_s2_comb_nonpre"
        bash run_train2.sh train_s3.py s3_s2_comb_nonpre --train_rev_v2c --num_epochs 300
    
    elif [[ $arg -eq 2 ]]; then
        echo "Input is equal to 2. Running s3_s2_comb_nonpre_s02"
        bash run_train2.sh train_s3.py s3_s2_comb_nonpre_s02 --train_rev_v2c --num_epochs 300 --subj_id 02
    
    elif [[ $arg -eq 3 ]]; then
        echo "Input is equal to 3. Running s3_s2_comb_nonpre_s03"
        bash run_train2.sh train_s3.py s3_s2_comb_nonpre_s03 --train_rev_v2c --num_epochs 300 --subj_id 03
    
    elif [[ $arg -eq 4 ]]; then
        echo "Input is equal to 4. Running s3_s2_comb_nonpre_s04"
        bash run_train2.sh train_s3.py s3_s2_comb_nonpre_s04 --train_rev_v2c --num_epochs 300 --subj_id 04

    else
        echo "Invalid input. Ignoring argument $arg."
    fi
done