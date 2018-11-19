#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem=30g
#SBATCH -t 0

#module load cuda-8.0 cudnn-8.0-5.1
source activate p27t041
python active_cls.py --acquiremethod lm_parallel --dataset mareview --softmax_type with_log --softmax_temp 2 > logs/mar_lm_withlog_t2.txt
