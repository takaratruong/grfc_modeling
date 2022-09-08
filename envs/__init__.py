from gym.envs.registration import load_env_plugins as _load_env_plugins
from gym.envs.registration import make, register, registry, spec


register(
    id="Humanoid_Treadmill",
    entry_point="envs.humanoid:HumanoidTreadmillEnv",
)


register(
    id="Skeleton",
    entry_point="envs.skeleton:SkeletonEnv",
)


register(
    id="Rajagopal",
    entry_point="envs.rajagopal:RajagopalEnv",
)

