#!/bin/bash
#SBATCH -J ITS
#SBATCH -t 3600
#SBATCH -o ./integrating_topics_syntax.%j.out
#SBATCH -e ./integrating_topics_syntax.%j.err
#SBATCH --mail-type=END
#SBATCH --mem=1G

module purge
module load anaconda3/latest

. $ANACONDA_HOME/etc/profile.d/conda.sh
conda activate integrating_topics_syntax
python src/evaluate.py --alpha=0.03 --beta=0.5 --gamma=0.1 --delta=0.1 --num_iter=6000 --num_topics=10 --num_classes=8 --dataset=data2000 --iteration=4800
conda deactivate