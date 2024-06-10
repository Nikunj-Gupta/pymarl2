import os 


configs=["iql", "vdn", "qmix", "coma", "cw_qmix", "dop", "ippo", "lica", "ow_qmix", "qatten", "qmix_att", "qmix_large", "qplex", "qtran", "riit", "riit_online", "vmix"] 
configs=["iql", "qmix", "vdn", "cw_qmix", "ow_qmix", "qmix_att", "qplex", "qtran"] 
configs=["iql", "qmix", "vdn", "qplex", "qtran"] 
configs=["vdn"] 
maps = ["aloha", "disperse", "gather", "hallway", "pursuit"] 
agents = ["gat", "gatv2", "gcn"] 
GPU = 1 
seed_max=1 

parallel = True 

for map in maps: 
    for config in configs: 
        for agent in agents: 
            for _ in range(seed_max): 
                command = f"CUDA_VISIBLE_DEVICES={GPU} python3 src/main.py --config={config} --env-config={map} with agent={agent} cg_edges=full"  
                if parallel: command += " &"
                os.system(command) 
