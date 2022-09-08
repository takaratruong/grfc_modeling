import configargparse

p = configargparse.ArgParser()

""" Logistics (Naming/Loading/Saving/etc..) """
p.add('-c', '--config', required=True, is_config_file=True, help='config file path')
p.add('-n', '--exp_name', type=str, default='default_exp_name')

# Wandb
p.add('--wandb',  action='store_true')
p.add('--project', type=str, required=True)

# Environment and Learning
p.add('--alg', type=str, default='ppo', help='ppo or amp')

p.add('--env_id', type=str, required=True)

p.add('--xml_file', type=str, required=True)
p.add('--num_envs', type=int, default=50)
p.add('--frame_skip', type=int, default=5)

p.add('--treadmill_velocity', type=float, default=1.25)
p.add('--max_ep_time', type=float, default=5.0)
p.add('--num_steps', type=int, default=100)

# Action and Rewards
p.add('--phase_action_mag', type=float, default=.01)

# Gait
p.add('--gait_ref_path', type=str, required=True)
p.add('--muscle_ref_path', type=str, default=None)
p.add('--gait_cycle_time', type=float, required=True)

# Models
p.add('--agent_hidden_layers', type=int, action='append')
p.add('--disc_hidden_layers', type=int, action='append')

p.add('--agent_obs_size', type=int, required=True)
p.add('--disc_obs_size', type=int, default=-1)

# Logging
p.add('--vid_rec_freq', type=int, default=100)


def load_args():
    args = p.parse_args()
    return args

if __name__ == "__main__":
    options = p.parse_args()

    print(options)
    print("----------")
    print(p.format_help())
    print("----------")
    print(p.format_values())