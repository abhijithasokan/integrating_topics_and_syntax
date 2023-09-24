#!/bin/bash
#SBATCH -J ITS
#SBATCH -t 3600
#SBATCH -n 24
#SBATCH -o ./its_para.%j.out
#SBATCH -e ./its_para.%j.err
#SBATCH --mail-type=END
#SBATCH --mem=1G

module purge
module load anaconda3/latest

. $ANACONDA_HOME/etc/profile.d/conda.sh
conda activate integrating_topics_syntax
python parallel_main.py --alpha=0.05 --beta=0.05 --gamma=0.05 --delta=0.05  --num_topics=10 --num_classes=8 --num_iter=6000 --burn_period=2000 --batch_size=256 --batch_iterations=100 --dataset=data11264
conda deactivate