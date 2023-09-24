#!/bin/bash
#SBATCH -J ITS
#SBATCH -t 7200
#SBATCH -o ./its_gen.%j.out
#SBATCH -e ./its_gen.%j.err
#SBATCH --mail-type=END
#SBATCH --mem=1024

module purge
module load anaconda3/latest

. $ANACONDA_HOME/etc/profile.d/conda.sh
conda activate integrating_topics_syntax
python generate.py --alpha=0.02 --beta=0.02 --gamma=0.02 --delta=0.02 --num_iter=6000 --iteration=2400 --dataset=data2000 --num_topics=10 --num_classes=8
conda deactivate
