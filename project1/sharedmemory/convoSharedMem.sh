#!/bin/bash

#SBATCH --job-name=convoSharedMem
#SBATCH --output=convoSharedMem.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=0

./convoSharedMem ../images/butterfly.jpg 10
./convoSharedMem ../images/car.jpg 10
./convoSharedMem ../images/cat.jpg 10
./convoSharedMem ../images/city.jpg 10
./convoSharedMem ../images/control.jpg 10
./convoSharedMem ../images/lizard.jpg 10
./convoSharedMem ../images/paisaje.jpg 10
./convoSharedMem ../images/planet.jpg 10
./convoSharedMem ../images/thunder.jpg 10
./convoSharedMem ../images/wood.jpg 10
