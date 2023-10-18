#!/bin/bash
sbatch <<EOT
#!/bin/bash
# Example of running python script in a batch mode 
#SBATCH -c 1                       # Number of CPU Cores 
#SBATCH -p gpus                    # Partition (queue) 
#SBATCH --gres gpu:1               # gpu:n, where n = number of GPUs 
#SBATCH --mem 32G                  # memory pool for all cores 
#SBATCH --job-name="$1Z$2L$3H$4"
#SBATCH --output=slurm.%N.%j.log   # Standard output and error loga

echo $1, $2 $3 $4


# Source Virtual environment (conda)
. /vol/biomedic2/agk21/anaconda3/etc/profile.d/conda.sh

conda activate cosa

./training.sh $1 $2 $3 $4 
EOT