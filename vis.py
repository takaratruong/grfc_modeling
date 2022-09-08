import gym
import envs # GYM CANT FIND IT OTHERWISE
import wandb
from algs.amp_ppo import RL
from configs.config_loader import load_args
import time
import numpy as np
from algs.amp_models import ActorCriticNet
import torch


if __name__ == '__main__':
    args = load_args()
    env = gym.vector.make(args.env_id, num_envs=1, args=args, new_step_api=True, render_mode='human')
    # env = gym.vector.make(args.env_id, num_envs=1, args=args, new_step_api=True)

    model = ActorCriticNet(env.single_observation_space.shape[0], env.single_action_space.shape[0], [256, 256])

    policy_path = 'results/models/humanoid_baseline/humanoid_baseline_iter1200.pt'

    model.load_state_dict(torch.load(policy_path))  # relative file path
    model.cuda()

    data = None
    state = env.reset()
    for i in range(20000):
        if i % 1000 == 0:
            print(i)

        with torch.no_grad():
            act = model.sample_best_actions(torch.from_numpy(state).cuda().type(torch.cuda.FloatTensor)).cpu().numpy()
        next_state, reward, terminated, done, info = env.step(act)

        state = next_state
        data = info['grfc_info'][0] if data is None else np.vstack((data, info['grfc_info'][0]))
        # print(data.shape)

    # np.save('grfc_data', data)


    # state = env.reset()
    # for i in range(5000):
    #     # print(i)
    #     act = env.action_space.sample()*0.01
    #     next_state, reward, terminated, done, info = env.step(act)
    #     # print(info['grfc_info'])
    #     # print(state)
    #     # print(next_state)
    #     # print('term', terminated)
    #     # print('done', done)
    #     # print(info)
    #     print()
    #
    #     state = next_state