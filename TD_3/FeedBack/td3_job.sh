#!/bin/sh

#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=1:mem=90gb:ngpus=1

module load lang/python/anaconda/pytorch

cd '/home/px19783/Two_joint_arm'

python < TD_3/TD3_FB_main.py
