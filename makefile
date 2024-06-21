GPU=6
all: 
	clear 
	# CUDA_VISIBLE_DEVICES=${GPU} python src/main.py --config=qmix --env-config=hallway with agent=gcn use_cuda=True cg_edges=full
	CUDA_VISIBLE_DEVICES=${GPU} python src/main.py --config=qmix --env-config=sc2_gen_protoss with agent=gcn cg_edges=full env_args.capability_config.n_units=10 env_args.capability_config.n_enemies=10 seed=0