HPS=clevr_hans3
MODEL=SA
./slurm_training.sh $MODEL gauss 0 $HPS
./slurm_training.sh $MODEL gauss 1 $HPS


MODEL=ASA
./slurm_training.sh $MODEL gauss 0 $HPS
./slurm_training.sh $MODEL gauss 1 $HPS


MODEL=VSA
./slurm_training.sh $MODEL gauss 0 $HPS
./slurm_training.sh $MODEL gauss 1 $HPS
./slurm_training.sh $MODEL gmm 0 $HPS
./slurm_training.sh $MODEL gmm 1 $HPS

MODEL=VASA
./slurm_training.sh $MODEL gauss 0 $HPS
./slurm_training.sh $MODEL gauss 1 $HPS
./slurm_training.sh $MODEL gmm 0 $HPS
./slurm_training.sh $MODEL gmm 1 $HPS

MODEL=SSA
./slurm_training.sh $MODEL gauss 0 $HPS
./slurm_training.sh $MODEL gauss 1 $HPS
./slurm_training.sh $MODEL gmm 0 $HPS
./slurm_training.sh $MODEL gmm 1 $HPS

MODEL=SSAU
./slurm_training.sh $MODEL gauss 0 $HPS
./slurm_training.sh $MODEL gauss 1 $HPS
./slurm_training.sh $MODEL gmm 0 $HPS
./slurm_training.sh $MODEL gmm 1 $HPS
# =============================================

HPS=bitmoji
MODEL=SA
./slurm_training.sh $MODEL gauss 0 $HPS
./slurm_training.sh $MODEL gauss 1 $HPS


MODEL=ASA
./slurm_training.sh $MODEL gauss 0 $HPS
./slurm_training.sh $MODEL gauss 1 $HPS


MODEL=VSA
./slurm_training.sh $MODEL gauss 0 $HPS
./slurm_training.sh $MODEL gauss 1 $HPS
./slurm_training.sh $MODEL gmm 0 $HPS
./slurm_training.sh $MODEL gmm 1 $HPS

MODEL=VASA
./slurm_training.sh $MODEL gauss 0 $HPS
./slurm_training.sh $MODEL gauss 1 $HPS
./slurm_training.sh $MODEL gmm 0 $HPS
./slurm_training.sh $MODEL gmm 1 $HPS

MODEL=SSA
./slurm_training.sh $MODEL gauss 0 $HPS
./slurm_training.sh $MODEL gauss 1 $HPS
./slurm_training.sh $MODEL gmm 0 $HPS
./slurm_training.sh $MODEL gmm 1 $HPS


# =============================================

HPS=objects_room
MODEL=SA
./slurm_training.sh $MODEL gauss 0 $HPS
./slurm_training.sh $MODEL gauss 1 $HPS


MODEL=ASA
./slurm_training.sh $MODEL gauss 0 $HPS
./slurm_training.sh $MODEL gauss 1 $HPS


MODEL=VSA
./slurm_training.sh $MODEL gauss 0 $HPS
./slurm_training.sh $MODEL gauss 1 $HPS
./slurm_training.sh $MODEL gmm 0 $HPS
./slurm_training.sh $MODEL gmm 1 $HPS

MODEL=VASA
./slurm_training.sh $MODEL gauss 0 $HPS
./slurm_training.sh $MODEL gauss 1 $HPS
./slurm_training.sh $MODEL gmm 0 $HPS
./slurm_training.sh $MODEL gmm 1 $HPS

MODEL=SSA
./slurm_training.sh $MODEL gauss 0 $HPS
./slurm_training.sh $MODEL gauss 1 $HPS
./slurm_training.sh $MODEL gmm 0 $HPS
./slurm_training.sh $MODEL gmm 1 $HPS


# =============================================

HPS=ffhq
MODEL=SA
./slurm_training.sh $MODEL gauss 0 $HPS
./slurm_training.sh $MODEL gauss 1 $HPS


MODEL=ASA
./slurm_training.sh $MODEL gauss 0 $HPS
./slurm_training.sh $MODEL gauss 1 $HPS


MODEL=VSA
./slurm_training.sh $MODEL gauss 0 $HPS
./slurm_training.sh $MODEL gauss 1 $HPS
./slurm_training.sh $MODEL gmm 0 $HPS
./slurm_training.sh $MODEL gmm 1 $HPS

MODEL=VASA
./slurm_training.sh $MODEL gauss 0 $HPS
./slurm_training.sh $MODEL gauss 1 $HPS
./slurm_training.sh $MODEL gmm 0 $HPS
./slurm_training.sh $MODEL gmm 1 $HPS

MODEL=SSA
./slurm_training.sh $MODEL gauss 0 $HPS
./slurm_training.sh $MODEL gauss 1 $HPS
./slurm_training.sh $MODEL gmm 0 $HPS
./slurm_training.sh $MODEL gmm 1 $HPS


# =============================================



