#!/bin/sh

#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=1:mem=10gb

module load lang/python/anaconda/pytorch

cd '/home/px19783/Two_joint_arm'

python < Supervised_learning/Feed_Forward/NN_Spvsd_FF_main.py