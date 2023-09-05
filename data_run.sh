#!/bin/bash
#SBATCH -J ITS
#SBATCH -n 1
#SBATCH -t 3600
#SBATCH -o ./integrating_topics_syntax.%j.out
#SBATCH -e ./integrating_topics_syntax.%j.err
#SBATCH --mail-type=END
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=64000
#SBATCH --partition=informatik-mind

module purge
module load anaconda3/latest

. $ANACONDA_HOME/etc/profile.d/conda.sh
conda activate integrating_topics_syntax
python preprocess_data.py
conda deactivate