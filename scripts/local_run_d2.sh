#!/bin/bash

# tetrominoes, clevr_tex, objects_room, multidsprites, clevr
for HPS in clevr #_tex tetrominoes objects_room multidsprites
do
    LOGSDIR="/vol/biomedic3/agk21/CoSA/SAModelling/EMLOGS-RandomSeeds/$HPS"

    for MODEL in VAE
    do
        for RUN in 1 2 3
        do
        
            RUNCMD="/vol/biomedic3/agk21/CoSA/SAModelling/main.py \
                    --exp_name $MODEL-shared-noch-highsd-init \
                    --model $MODEL \
                    --hps $HPS \
                    --ckpt_dir $LOGSDIR \
                    --x_like shared_dgauss \
                    --std_init 1.0 \
                    --learning_rate 0.001 \
                    --run_idx $RUN \
                    --learn_prior"
            echo $RUNCMD
            CUDA_VISIBLE_DEVICES=0 python $RUNCMD

            #
            # RUNCMD="/vol/biomedic3/agk21/CoSA/SAModelling/main.py \
            #         --exp_name EM-fixed-$MODEL \
            #         --model $MODEL \
            #         --hps $HPS \
            #         --ckpt_dir $LOGSDIR \
            #         --EM_slots yes_fixed \
            #         --x_like mse \
            #         --run_idx $RUN"
            # echo $RUNCMD
            # CUDA_VISIBLE_DEVICES=1 python $RUNCMD

            # RUNCMD="/vol/biomedic3/agk21/CoSA/SAModelling/main.py \
            #         --exp_name EM-dynamic-$MODEL \
            #         --model $MODEL \
            #         --hps $HPS \
            #         --ckpt_dir $LOGSDIR \
            #         --EM_slots yes_dynamic \
            #         --x_like mse \
            #         --run_idx $RUN"
            # echo $RUNCMD
            # CUDA_VISIBLE_DEVICES=0 python $RUNCMD
        done
        wait
    done
done