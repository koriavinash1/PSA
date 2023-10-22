#!/bin/bash

for RUN in 1 2 3 4 5
    for MODEL in SA ASA VSA VASA SSA
    do
        for HPS in clevr bitmoji ffhq objects_room
        do 

            # ./slurm_training.sh $MODEL gauss 0 $HPS $RUN
            # ./slurm_training.sh $MODEL gauss 1 $HPS $RUN
            ./slurm_training.sh $MODEL gmm 0 $HPS $RUN
            ./slurm_training.sh $MODEL gmm 1 $HPS $RUN
        done
    done
done 
