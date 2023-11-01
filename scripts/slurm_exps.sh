#!/bin/bash

for HPS in clevr # shapestacks tetrominoes multidsprites objects_room
do
    for RUN in 1 2 3 4 5
    do
        for MODEL in VSA VASA SSA
        do 
            # ./slurm_training.sh $MODEL gauss 1 $HPS $RUN
            ./slurm_training.sh $MODEL gmm 1 $HPS $RUN
        done

        for MODEL in SA ASA
        do
            ./slurm_training.sh $MODEL gauss 1 $HPS $RUN
        done
    done
done 
