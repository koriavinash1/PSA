#!/bin/bash

# tetrominoes, clevr_tex, objects_room, multidsprites, clevr
for HPS in clevr #_tex tetrominoes objects_room multidsprites
do
    LOGSDIR="/vol/biomedic3/agk21/CoSA/SAModelling/EMLOGS-RandomSeeds/$HPS"

    for MODEL in SA
    do
        for RUN in 1 # 2 3
        do
            # RUNCMD="/vol/biomedic3/agk21/CoSA/SAModelling/main.py \
            #         --exp_name $MODEL \
            #         --model $MODEL \
            #         --hps $HPS \
            #         --ckpt_dir $LOGSDIR \
            #         --x_like mse \
            #         --run_idx $RUN"
            # echo $RUNCMD
            # CUDA_VISIBLE_DEVICES=0 python $RUNCMD

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

            RUNCMD="/vol/biomedic3/agk21/CoSA/SAModelling/main.py \
                    --exp_name EM-fixed-mle-xnll-s1.0-$MODEL \
                    --model $MODEL \
                    --hps $HPS \
                    --ckpt_dir $LOGSDIR \
                    --EM_slots yes_fixed_mle \
                    --x_like fixed_dgauss \
                    --std_init 1.0 \
                    --run_idx $RUN \
                    --learn_prior"
            echo $RUNCMD
            CUDA_VISIBLE_DEVICES=0 python $RUNCMD
        done
        wait
    done
done