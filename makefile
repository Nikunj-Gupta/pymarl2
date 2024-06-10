GPU=0
all: 
	clear 
	CUDA_VISIBLE_DEVICES=${GPU} python src/main.py --config=qmix_large --env-config=hallway with agent=gcn cg_edges=cycle 