#!/bin/bash -l
# Set SCC project
#$ -P ece601
#$ -N BUGenerator_train

# Merge stderr and stdout
#$ -j y

# Request 1 CPU
#$ -pe omp 2

# Time limit 3hrs
#$ -l h_rt=3:00:00

# Request 1 GPU (the number of GPUs needed should be divided by the number of CPUs requested above)
#$ -l gpus=0.5

# Specify the minimum GPU compute capability
#$ -l gpu_c=3.5

# module load python/3.6.2 cuda/8.0 cudnn/5.1 tensorflow

# Per INC12572943 at BU ITHC, cudnn/7.2 is included in cuda/9.2 now
module load python/3.6.0 cuda/9.2 tensorflow/r1.10 gcc/6.2.0
# module load python/3.6.2 cuda/9.1 cudnn/7.1 tensorflow/r1.8 gcc/6.2.0
pip install --user keras
cd /projectnb/ece601/BUGenerator/Model/python

# program name or command and its options and arguments
python validate.py