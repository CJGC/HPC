#!/bin/bash

#SBATCH --job-name=convoConstMem
#SBATCH --output=convoConstMem.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=0

./convoConstMem ../images/butterfly.jpg 10
./convoConstMem ../images/car.jpg 10
./convoConstMem ../images/cat.jpg 10
./convoConstMem ../images/city.jpg 10
./convoConstMem ../images/control.jpg 10
./convoConstMem ../images/lizard.jpg 10
./convoConstMem ../images/paisaje.jpg 10
./convoConstMem ../images/planet.jpg 10
./convoConstMem ../images/thunder.jpg 10
./convoConstMem ../images/wood.jpg 10
