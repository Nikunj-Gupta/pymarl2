GPU=0 
all: 
	clear 
	# CUDA_VISIBLE_DEVICES=${GPU} python src/main.py --config=qmix --env-config=hallway with agent=gcn use_cuda=True cg_edges=full
	# CUDA_VISIBLE_DEVICES=${GPU} python src/main.py --config=qmix --env-config=sc2_gen_protoss with agent=gcn cg_edges=full env_args.capability_config.n_units=10 env_args.capability_config.n_enemies=10 seed=0
	# CUDA_VISIBLE_DEVICES=${GPU} python src/main.py --config=ow_qmix --env-config=gather with agent=rnn seed=0 use_cuda=False 
	# CUDA_VISIBLE_DEVICES=${GPU} python src/main.py --config=ow_qmix --env-config=gather with agent=n_rnn seed=0 use_cuda=False 
	# CUDA_VISIBLE_DEVICES=${GPU} python src/main.py --config=ow_qmix --env-config=gather with agent=gtn seed=0 
	# CUDA_VISIBLE_DEVICES=${GPU} python src/main.py --config=ow_qmix --env-config=gather with agent=gtn cg_edges=allstar 

gtn: 
	CUDA_VISIBLE_DEVICES=${GPU} python src/main.py --config=ow_qmix --env-config=gather with agent=rnn seed=0 use_cuda=False t_max=5000000 
	CUDA_VISIBLE_DEVICES=${GPU} python src/main.py --config=ow_qmix --env-config=gather with agent=n_rnn seed=0 use_cuda=False t_max=5000000 
	CUDA_VISIBLE_DEVICES=${GPU} python src/main.py --config=ow_qmix --env-config=gather with agent=gtn seed=0 cg_edges=lcs t_max=5000000 
	CUDA_VISIBLE_DEVICES=${GPU} python src/main.py --config=ow_qmix --env-config=gather with agent=gtn seed=0 cg_edges=allstar t_max=5000000 

	CUDA_VISIBLE_DEVICES=${GPU} python src/main.py --config=ow_qmix --env-config=gather with agent=rnn seed=1 t_max=5000000 use_cuda=False 
	CUDA_VISIBLE_DEVICES=${GPU} python src/main.py --config=ow_qmix --env-config=gather with agent=n_rnn seed=1 t_max=5000000 use_cuda=False 
	CUDA_VISIBLE_DEVICES=${GPU} python src/main.py --config=ow_qmix --env-config=gather with agent=gtn seed=1 cg_edges=lcs t_max=5000000 
	CUDA_VISIBLE_DEVICES=${GPU} python src/main.py --config=ow_qmix --env-config=gather with agent=gtn seed=1 cg_edges=allstar t_max=5000000 

	CUDA_VISIBLE_DEVICES=${GPU} python src/main.py --config=ow_qmix --env-config=gather with agent=rnn seed=2 t_max=5000000 use_cuda=False 
	CUDA_VISIBLE_DEVICES=${GPU} python src/main.py --config=ow_qmix --env-config=gather with agent=n_rnn seed=2 t_max=5000000 use_cuda=False 
	CUDA_VISIBLE_DEVICES=${GPU} python src/main.py --config=ow_qmix --env-config=gather with agent=gtn seed=2 cg_edges=lcs t_max=5000000 
	CUDA_VISIBLE_DEVICES=${GPU} python src/main.py --config=ow_qmix --env-config=gather with agent=gtn seed=2 cg_edges=allstar t_max=5000000 

