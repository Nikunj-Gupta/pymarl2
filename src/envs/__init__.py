from functools import partial
import sys
import os
from smacv2.env import StarCraft2Env, StarCraftCapabilityEnvWrapper
from .multiagentenv import MultiAgentEnv 

from .starcraft import StarCraft2Env
from .matrix_game import OneStepMatrixGame
from .stag_hunt import StagHunt 
from .maco.aloha import AlohaEnv
from .maco.pursuit import PursuitEnv
from .maco.sensors import SensorEnv
from .maco.hallway import HallwayEnv
from .maco.disperse import DisperseEnv
from .maco.gather import GatherEnv

try:
    gfootball = True
    from .gfootball import GoogleFootballEnv
except Exception as e:
    gfootball = False
    print(e)

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["sc2wrapped"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper)
REGISTRY["aloha"] = partial(env_fn, env=AlohaEnv)
REGISTRY["pursuit"] = partial(env_fn, env=PursuitEnv)
REGISTRY["sensor"] = partial(env_fn, env=SensorEnv)
REGISTRY["hallway"] = partial(env_fn, env=HallwayEnv)
REGISTRY["disperse"] = partial(env_fn, env=DisperseEnv)
REGISTRY["gather"] = partial(env_fn, env=GatherEnv)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)

if gfootball:
    REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
