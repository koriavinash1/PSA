#!/bin/bash

# tetrominoes, clevr_tex, objects_room, multidsprites, clevr

for HPS in clevr #_tex tetrominoes objects_room multidsprites
do
    LOGSDIR="/vol/biomedic3/agk21/CoSA/SAModelling/LOGS-RandomSeeds/$HPS"

    for MODEL in VSA VASA SSA
    do
        for RUN in 1 2 3 4 5
        do
            RUNCMD="/vol/biomedic3/agk21/CoSA/SAModelling/main.py \
                    --exp_name $MODEL'LGMM' \
                    --learn_prior \
                    --zprior gmm \
                    --model $MODEL \
                    --hps $HPS \
                    --ckpt_dir $LOGSDIR \
                    --run_idx $RUN"
            echo $RUNCMD
            if [ "$MODEL" = "SSA" ]; then
                CUDA_VISIBLE_DEVICES=0 python $RUNCMD
            else
                CUDA_VISIBLE_DEVICES=0 python $RUNCMD
            fi
        done
    done
    
    for MODEL in SA ASA
    do
        for RUN in 1 2 3 4 5
        do
            RUNCMD="/vol/biomedic3/agk21/CoSA/SAModelling/main.py \
                    --exp_name $MODEL \
                    --model $MODEL \
                    --hps $HPS \
                    --ckpt_dir $LOGSDIR \
                    --run_idx $RUN"
            echo $RUNCMD
            CUDA_VISIBLE_DEVICES=0 python $RUNCMD
        done
    done

done


