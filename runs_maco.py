import os 

def baselines(): 
    CONFIGS = ["qmix", "vdn", "iql", "qtran", "qatten", "qplex", "cw_qmix", "ow_qmix"]
    ENVS = ["gather", "hallway", "pursuit", "disperse", "sensor", "aloha"] 
    SEEDS = 3 
    PARALLEL = False 

    for s in range(SEEDS): 
        for e in ENVS:
            for a in CONFIGS: 
                command = f"python3 src/main.py --config={a} --env-config={e} with use_cuda=False seed={s}" 
                if PARALLEL: command += " &" 
                os.system(command) 
baselines() 

# CONFIGS = ["qmix", "vdn", "iql", "qtran", "qatten", "qplex", "cw_qmix", "ow_qmix"]
# CONFIGS = ["qmix", "vdn", "cw_qmix", "ow_qmix"]
# ENVS = ["gather", "hallway", "pursuit", "disperse", "sensor", "aloha"] 
# GNN_QMIX = ["gcn", "gat", "gatv2"] 
# CG_EDGES = ["full", "line", "cycle", "star"] 
# SEEDS = 3 

# """
# GTN-cgedges-new
# """
# for s in range(SEEDS): 
#     for e in ENVS:
#         for cg_edges in CG_EDGES: 
#             command = f"CUDA_VISIBLE_DEVICES=2 python3 src/main.py --config=ow_qmix --env-config={e} with agent=gtn cg_edges={cg_edges} use_cuda=True seed={s}" 
#             if PARALLEL: command += " &" 
#             os.system(command) 

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
#         # command = f"CUDA_VISIBLE_DEVICES=5 python3 src/main.py --config=qmix --env-config={e} with agent=gtn use_cuda=True seed={s} cg_edges=fff" 
#         # command = f"CUDA_VISIBLE_DEVICES=4 python3 src/main.py --config=qmix --env-config={e} with agent=gtn use_cuda=True seed={s} cg_edges=fffff" 
#         command = f"CUDA_VISIBLE_DEVICES=3 python3 src/main.py --config=qmix --env-config={e} with agent=gtn use_cuda=True seed={s} cg_edges=ffffffffff" 
#         if PARALLEL: command += " &" 
#         os.system(command) 