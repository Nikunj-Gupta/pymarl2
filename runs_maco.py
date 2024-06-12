import os 


# configs=["iql", "vdn", "qmix", "coma", "cw_qmix", "dop", "ippo", "lica", "ow_qmix", "qatten", "qmix_att", "qmix_large", "qplex", "qtran", "riit", "riit_online", "vmix"] 
# maps = ["aloha", "disperse", "gather", "hallway", "pursuit"] 
# agents = ["gat", "gatv2", "gcn", "gtn"] 

configs=["qmix", "vdn"] 
maps = ["aloha", "disperse", "gather", "hallway", "pursuit"] 
agents = ["gtn"] 
GPU = 0 
seed_max=1 

parallel = True 

# for map in maps: 
#     for config in configs: 
#         for agent in agents: 
#             for _ in range(seed_max): 
#                 command = f"CUDA_VISIBLE_DEVICES={GPU} python3 src/main.py --config={config} --env-config={map} with agent={agent} cg_edges=full"  
#                 if parallel: command += " &"
#                 os.system(command) 
for map in maps: 
    for config in configs: 
        for _ in range(seed_max): 
            command = f"CUDA_VISIBLE_DEVICES={GPU} python3 src/main.py --config={config} --env-config={map} with cg_edges=full" 
            if parallel: command += " &"
            os.system(command) 
