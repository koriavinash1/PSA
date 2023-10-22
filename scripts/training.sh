#!/bin/bash
# Run python script

# $1 slot_arch $2 hps
MODEL=$1
ZPRIOR=$2
LEARNZPRIOR=$3
HPS=$4
RUN=$5
BASENAME="PSA"
EXPNAME="-$ZPRIOR-$MODEL"

LOGSDIR="/vol/biomedic3/agk21/CoSA/SAModelling/LOGS/$HPS"

RUNCMD="/vol/biomedic3/agk21/CoSA/SAModelling/main.py \
        --ckpt_dir $LOGSDIR \
        --hps $HPS \
        --model $MODEL \
        --zprior $ZPRIOR \
        --run_idx $RUN"


if [ "$LEARNZPRIOR" = "1" ]; then
    EXPNAME="$EXPNAME-LearnZprior"
    RUNCMD="$RUNCMD --learn_prior"
fi


BASENAME="$BASENAME-$EXPNAME"

RUNCMD="$RUNCMD --exp_name $BASENAME"

echo "$RUNCMD"
python $RUNCMD