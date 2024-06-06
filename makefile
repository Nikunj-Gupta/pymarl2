GPU=0
all: 
	clear 
	CUDA_VISIBLE_DEVICES=${GPU} python src/main.py --config=qplex --env-config=hallway 