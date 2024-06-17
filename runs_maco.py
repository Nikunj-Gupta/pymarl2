import os 
PARALLEL = False 

"""
baselines 
"""
def baselines(maps, configs, gpu, seed_max): 
    for map in maps: 
        for config in configs: 
            for _ in range(seed_max): 
                command = f"CUDA_VISIBLE_DEVICES={gpu} python3 src/main.py --config={config} --env-config={map} " 
                if PARALLEL: command += " &"
                print(command) 
                os.system(command) 
# baselines(
#     maps = ["aloha", "disperse", "gather", "hallway", "pursuit", "sensor"], 
#     configs=["iql", "vdn", "qmix", "cw_qmix", "dop", "ow_qmix", "qatten", "qmix_att", "qplex", "qtran"], 
#     gpu = 0, 
#     seed_max=1 
# )

"""
cg-marl 
"""
def cgmarl(maps, configs, agents, cg_edges, gpu, seed_max): 
    for map in maps: 
        for config in configs: 
            for agent in agents: 
                for cg_edge in cg_edges: 
                    for _ in range(seed_max): 
                        command = f"CUDA_VISIBLE_DEVICES={gpu} python3 src/main.py --config={config} --env-config={map} with agent={agent} cg_edges={cg_edge} "                         
                        if PARALLEL: command += " &"
                        print(command) 
                        # os.system(command) 
# cgmarl(
#     maps = ["aloha", "disperse", "gather", "hallway", "pursuit", "sensor"], 
#     configs=["vdn", "qmix"], 
#     agents = ["gcn", "gat", "gatv2", "gtn"], 
#     cg_edges = ["full", "star", "cycle", "line"], 
#     GPU=0, 
#     seed_max=1 
# )

"""
DMCG 
"""
def dmcg(maps, configs, agents, gpu, seed_max): 
    for map in maps: 
        for config in configs: 
            for agent in agents: 
                for _ in range(seed_max): 
                    command = f"CUDA_VISIBLE_DEVICES={gpu} python3 src/main.py --config={config} --env-config={map} with agent={agent} "                         
                    if PARALLEL: command += " &"
                    print(command) 
                    os.system(command) 
# dmcg(
#     maps = ["sensor"], 
#     configs=["qmix"], 
#     agents = ["gtn"], 
#     gpu=0, 
#     seed_max=1 
# )

baselines(
    maps = ["sensor"], 
    configs=["qmix", "iql", "vdn"], 
    gpu = 0, 
    seed_max=1 
)
