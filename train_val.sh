#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_Synapse_MagicNet.py --seed 1337 --labelnum 4
CUDA_VISIBLE_DEVICES=0 python train_Synapse_CPS.py --seed 1337 --labelnum 4

CUDA_VISIBLE_DEVICES=0 python train_AMOS_MagicNet.py --seed 1337 --labelnum 10
CUDA_VISIBLE_DEVICES=0 python train_AMOS_CPS.py --seed 1337 --labelnum 10