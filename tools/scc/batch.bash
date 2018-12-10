#!/bin/bash -l
# Set SCC project
#$ -P ece601
#$ -N BUGenerator_train

# Request 1 CPU
#$ -pe omp 1

# Request 1 GPU (the number of GPUs needed should be divided by the number of CPUs requested above)
#$ -l gpus=1

# Specify the minimum GPU compute capability
#$ -l gpu_c=3.5

# module load python/3.6.2 cuda/8.0 cudnn/5.1 tensorflow
module load python/3.6.0 cuda/8.0 cudnn/5.1 tensorflow/r1.0_python-3.6.0
cd /projectnb/ece601/BUGenerator

# program name or command and its options and arguments
python Model/python/u-net-remote-train.py