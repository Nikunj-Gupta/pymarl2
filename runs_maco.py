import os 
PARALLEL = False 

CONFIGS = ["qmix", "vdn", "iql", "qtran", "qatten", "qplex", "cw_qmix", "ow_qmix"]
ENVS = ["gather", "hallway", "pursuit", "disperse", "sensor", "aloha"] 
GNN_QMIX = ["gcn", "gat", "gatv2"] 
CG_EDGES = ["full", "line", "cycle", "star"] 
SEEDS = 5 

# """
# Baselines 
# """
# for s in range(SEEDS): 
#     for e in ENVS:
#         for a in CONFIGS: 
#             command = f"CUDA_VISIBLE_DEVICES=6 python3 src/main.py --config={a} --env-config={e} with use_cuda=True seed={s}" 
#             if PARALLEL: command += " &" 
#             os.system(command) 

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

# """
# GTN-Qmix 
# """
# for s in range(SEEDS): 
#     for e in ENVS:
#         command = f"CUDA_VISIBLE_DEVICES=7 python3 src/main.py --config=qmix --env-config={e} with agent=gtn use_cuda=True seed={s}" 
#         if PARALLEL: command += " &" 
#         os.system(command) 