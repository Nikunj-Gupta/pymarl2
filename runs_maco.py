import os 


configs=["iql", "vdn", "qmix", "coma", "cw_qmix", "dop", "ippo", "lica", "ow_qmix", "qatten", "qmix_att", "qmix_large", "qplex", "qtran", "riit", "riit_online", "vmix"] 
configs=["cw_qmix", "ow_qmix", "qmix_att", "qplex", "qtran"] 
maps = ["aloha", "disperse", "gather", "hallway", "pursuit"] 
GPU = 0 
seed_max=1 

parallel = True 

for map in maps: 
    for config in configs: 
        for _ in range(seed_max): 
            command = f"CUDA_VISIBLE_DEVICES={GPU} python3 src/main.py --config={config} --env-config={map}" 
            if parallel: command += " &"
            os.system(command) 
