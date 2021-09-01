# From Laurence

import os
import sys
import argparse
import torch
import numpy as np
import subprocess

# use following arguments: -c 1 -t 2  --cmd <python MB_DPG/FeedForward/HyperParam_tuning/send_training.py

parser = argparse.ArgumentParser()
parser.add_argument('--cpus',    '-c', type=int, nargs='?', default=1)
parser.add_argument('--gpus',    '-g', type=int, nargs='?')
parser.add_argument('--hours',   '-t', type=int, nargs='?', default=23, help="time in hours")
parser.add_argument('--mins',          type=int, nargs='?', default=0, help="time in mins")
parser.add_argument('--memory',  '-m', type=int, nargs='?',             help="memory in gp")
parser.add_argument('--queue',         type=str, nargs='?',             help="queue")
parser.add_argument('--gputype',       type=str, nargs='?',             choices=["GTX1080", "RTX2080", "RTX2080Ti", "RTX3090"], help="{GTX1080, RTX2080, RTX2080Ti, RTX3090}")
parser.add_argument('--venv',          type=str, nargs='?')
parser.add_argument('--cmd',           type=str, nargs='*',             help="job command --- must be last argument")
parser.add_argument('--autoname', action='store_true',                   help="extract output filename based on job --- always uses third argument after --cmd, as in --cmd python file.py output_filename")
#parser.add_argument('ssd', '-s', action=store_true)


# split input args on --cmd
cmd_idx = sys.argv.index('--cmd')
args = sys.argv[1:cmd_idx]
args = parser.parse_args(args)
cmd = ' '.join(sys.argv[(1+cmd_idx):])

actor_ln = torch.tensor([7.52500026e-04])
critic_ln = torch.tensor([9.99999975e-06 ])




print(f"#!/bin/sh",)
print(f"#PBS -l walltime={args.hours}:{args.mins}:00")
select = f"#PBS -l select=1:ncpus={args.cpus}"

if args.memory is not None:
    select = select + f":mem={args.memory}gb"
if args.gpus is not None:
    select = select + f":ngpus={args.gpus}"
if args.gputype is not None:
    select = select + f":gputype={args.gputype}"

print(select)

if args.queue is not None:
    print(f"#PBS -q {args.queue}")

#print(f"#PBS -j oe") #Join stdout and stderr to stdout
if args.autoname:
    path = sys.argv[cmd_idx+3]
    print(f"#PBS -N {os.path.basename(path)}")
    print(f"#PBS -o /dev/null") #{path}.o")

print('')
print("cd", "'/home/px19783/Two_joint_arm'")

print('module load lang/python/anaconda/pytorch')
if args.venv is not None:
    print(f'source {args.venv}/bin/activate')
print('')

if args.autoname:
    #directly pipe output to file (so that its available immediately)
    print(f"{cmd} 2>&1 | tee {path}.o")
else:

     np.random.seed(1)
     seeds = np.random.choice([i for i in range(0, 1000) if i not in [37, 12, 72,  9, 75,35, 71, 33, 59, 61]],size=95)
     i = 1
     for seed in seeds:
        for crt in critic_ln:
            for act in actor_ln:

                print(cmd +" -s " + str(seed) + " -c " + str(crt.item()) + " -a " + str(act.item()) + " -i " + str(i))
                i +=1