import os 
PARALLEL = False 

ENVS = ["sc2_gen_protoss", "sc2_gen_terran", "sc2_gen_zerg"] 
UNITS = ["10v10", "5v5", "10v11", "20v20", "20v23"] 
CONFIGS = ["qmix", "vdn", "iql", "qtran", "qatten", "qplex", "cw_qmix", "ow_qmix"]
GNN_QMIX = ["gcn", "gat", "gatv2"] 
CG_EDGES = ["full", "line", "cycle", "star"] 
SEEDS = 3 


# """
# Baselines 
# """
# for s in range(SEEDS): 
#     for u in UNITS:
#         for e in ENVS:
#             for a in CONFIGS: 
#                 command = f"CUDA_VISIBLE_DEVICES=6 python3 src/main.py --config={a} --env-config={e} with use_cuda=True env_args.capability_config.n_units={u.split('v')[0]} env_args.capability_config.n_enemies={u.split('v')[1]} seed={s}" 
#                 if PARALLEL: command += " &" 
#                 os.system(command) 

# """
# GNN-Qmix 
# """
# for s in range(SEEDS): 
#     for cg_edges in CG_EDGES: 
#         for e in ENVS:
#             for a in GNN_QMIX: 
#                 command = f"CUDA_VISIBLE_DEVICES=6 python3 src/main.py --config=qmix --env-config={e} with agent={a} cg_edges={cg_edges} use_cuda=True seed={s}" 
#                 if PARALLEL: command += " &" 
#                 os.system(command) 

"""
GTN-Qmix 
"""
for s in range(SEEDS): 
    for u in UNITS:
        for e in ENVS:
            command = f"CUDA_VISIBLE_DEVICES=7 python3 src/main.py --config=qmix --env-config={e} with agent=gtn use_cuda=True env_args.capability_config.n_units={u.split('v')[0]} env_args.capability_config.n_enemies={u.split('v')[1]} seed={s}" 
            if PARALLEL: command += " &" 
            os.system(command) 