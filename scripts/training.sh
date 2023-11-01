#!/bin/bash
# Run python script

# $1 slot_arch $2 hps
MODEL=$1
ZPRIOR=$2
LEARNZPRIOR=$3
HPS=$4
RUN=$5
EXPNAME=$MODEL

LOGSDIR="/vol/biomedic3/agk21/CoSA/SAModelling/LOGS-RandomSeeds/$HPS"

RUNCMD="/vol/biomedic3/agk21/CoSA/SAModelling/main.py \
        --ckpt_dir $LOGSDIR \
        --hps $HPS \
        --model $MODEL \
        --zprior $ZPRIOR \
        --run_idx $RUN"


if [ "$LEARNZPRIOR" = "1" ]; then
    if [ "$ZPRIOR" = "gmm" ]; then
        EXPNAME=$EXPNAME'LGMM'
    fi
    RUNCMD="$RUNCMD --learn_prior"
fi




RUNCMD="$RUNCMD --exp_name $EXPNAME"

echo "$RUNCMD"
python $RUNCMD