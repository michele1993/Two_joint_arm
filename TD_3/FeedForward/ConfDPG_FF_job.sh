#!/bin/sh

#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=1:mem=10gb

module load lang/python/anaconda/pytorch

cd '/home/px19783/Two_joint_arm'

python < TD_3/FeedForward/Conf_FF_DPG_main.py