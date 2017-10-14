#!/bin/bash

#SBATCH --job-name=convoGlobalMem
#SBATCH --output=convoGlobalMem.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=0

./convoGlobalMem ../images/butterfly.jpg 20
./convoGlobalMem ../images/car.jpg 20
./convoGlobalMem ../images/cat.jpg 20
./convoGlobalMem ../images/city.jpg 20
./convoGlobalMem ../images/control.jpg 20
./convoGlobalMem ../images/lizard.jpg 20
./convoGlobalMem ../images/paisaje.jpg 20
./convoGlobalMem ../images/planet.jpg 20
./convoGlobalMem ../images/thunder.jpg 20
./convoGlobalMem ../images/wood.jpg 20
