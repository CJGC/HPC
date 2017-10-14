#!/bin/bash

#SBATCH --job-name=convoSharedMem
#SBATCH --output=convoSharedMem.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=0

./convoSharedMem ../images/butterfly.jpg 20
./convoSharedMem ../images/car.jpg 20
./convoSharedMem ../images/cat.jpg 20
./convoSharedMem ../images/city.jpg 20
./convoSharedMem ../images/control.jpg 20
./convoSharedMem ../images/lizard.jpg 20
./convoSharedMem ../images/paisaje.jpg 20
./convoSharedMem ../images/planet.jpg 20
./convoSharedMem ../images/thunder.jpg 20
./convoSharedMem ../images/wood.jpg 20
