import os
RUNS_DIRECTORY = "runs_discovery_baselines/" 
PARTITION = 

def write_run_file(content, num): 
    os.makedirs(RUNS_DIRECTORY, exist_ok=True)    
    f = open(f"runs_discovery/run_{num}.job", "a")
    f.write(content)
    f.close()

file = """#!/bin/bash

#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=./slurm_files/

eval "$(conda shell.bash hook)"
conda activate pymarl 
module load gcc/11.3.0 git/2.36.1 

""" 

# command = f"""
# python src/main.py --config={a} --env-config=gather with agent=gtn seed=0 use_cuda=False cg_edges=full
# """

# print(file+command)

CONFIGS = ["qmix", "vdn", "cw_qmix", "ow_qmix"]
ENVS = ["gather", "hallway", "pursuit", "disperse", "sensor", "aloha"] 
CG_EDGES = ["full", "line", "cycle", "star"] 
SEEDS = 1

"""
GTN-cgedges-new
"""
count = 0 
for s in range(SEEDS): 
    for e in ENVS:
        for a in CONFIGS:
            for cg_edges in CG_EDGES: 
                count+=1
                command = f"""python src/main.py --config={a} --env-config={e} with agent=gtn cg_edges={cg_edges} use_cuda=False seed={s}""" 
                write_run_file(file+command, count)