GPU=0
all: 
	clear 
	CUDA_VISIBLE_DEVICES=${GPU} python src/main.py --config=vdn --env-config=hallway with agent=gtn 