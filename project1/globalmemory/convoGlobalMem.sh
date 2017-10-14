#!/bin/bash

#SBATCH --job-name=convoGlobalMem
#SBATCH --output=convoGlobalMem.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=0

./convoGlobalMem ../images/butterfly.jpg 10
./convoGlobalMem ../images/car.jpg 10
./convoGlobalMem ../images/cat.jpg 10
./convoGlobalMem ../images/city.jpg 10
./convoGlobalMem ../images/control.jpg 10
./convoGlobalMem ../images/lizard.jpg 10
./convoGlobalMem ../images/paisaje.jpg 10
./convoGlobalMem ../images/planet.jpg 10
./convoGlobalMem ../images/thunder.jpg 10
./convoGlobalMem ../images/wood.jpg 10
