#!/bin/sh

#PBS -l walltime=00:01:00
#PBS -l select=1:ncpus=1:mem=1h0gb:ngpus=1

cd '/home/px19783'

module load lang/python/anaconda/pytorch

python < Two_joint_arm/TD_3/TD3_FB_main.py
