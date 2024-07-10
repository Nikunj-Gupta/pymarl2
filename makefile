all: 
	clear 
	python src/main.py --config=qmix --env-config=gather with seed=0 use_cuda=False 
	
request: 
	clear 
	salloc --partition=main --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=128G --time=1:00:00 

runall: 
	for f in runs_discovery/*.job; do sbatch $$f; done