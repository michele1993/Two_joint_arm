#!/bin/sh

#PBS -l walltime=00:01:00
#PBS -l select=1:ncpus=1:mem=10gb:ngpus=1

# cd ${PBS_O_WORKDIR}

module load lang/python/anaconda

time python < TD3_FB_main.py
