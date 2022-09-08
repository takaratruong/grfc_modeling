import gym
import envs # GYM CANT FIND IT OTHERWISE
import wandb
from algs.amp_ppo import RL
from configs.config_loader import load_args
import time
import numpy as np
from algs.amp_models import ActorCriticNet
import torch

from algs.amp_ppo import RL

if __name__ == '__main__':
    args = load_args()

    run_id = '/default'
    if args.wandb:
        run = wandb.init(project=args.project, config=args, name=args.exp_name, monitor_gym=True)
        run_id = f"/{run.id}"
        wandb.define_metric("step", hidden=True)
        wandb.define_metric("eval/reward", step_metric="step")
        wandb.define_metric("eval/ep_len", step_metric="step")

        wandb.define_metric("train/critic loss", step_metric="step")
        wandb.define_metric("train/actor loss", step_metric="step")
        wandb.define_metric("train/disc loss", step_metric="step")

    # Create a list of envs for training where the last one is also used to record videos
    envs = [lambda: gym.make(args.env_id, args=args, new_step_api=True) for _ in range(args.num_envs - 1)] + \
           [lambda: gym.wrappers.RecordVideo(gym.make(args.env_id, args=args, new_step_api=True, render_mode='rgb_array'), video_folder='results/videos' + run_id, name_prefix="rl-video", episode_trigger=lambda x: x % args.vid_rec_freq == 0, new_step_api=True)]

    # Vectorize environments w/ multi-processing
    envs = gym.vector.AsyncVectorEnv(envs, new_step_api=True, shared_memory=True)

    # Wrap to record ep rewards and ep lengths
    envs = gym.wrappers.RecordEpisodeStatistics(envs, new_step_api=True, deque_size=50)

    # Initialize RL and Train
    ppo = RL(envs, args)
    ppo.train()

    # Close
    envs.close()

    if args.wandb:
        run.finish()
