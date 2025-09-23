#!/bin/sh

#SBATCH --job-name=quietstar-train
#SBATCH --output=quietstar-train.out
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80gb
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hjcho@usc.edu

source ~/.bashrc
module purge
module load conda
module load cuda cudnn 

conda activate quietstar
cd /project2/jonmay_1426/hjcho/ntp_rl/quiet-star
python quiet-star-train.py --max_length 128  --lora --learning_rate 1e-4 --lora_r 64 --lora_alpha 128