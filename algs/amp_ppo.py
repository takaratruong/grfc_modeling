import torch
import numpy as np
import time
import torch.multiprocessing as mp
import pickle
import torch.optim as optim
import wandb
import os
from scipy.interpolate import interp1d
import ipdb
from copy import deepcopy
from algs.amp_models import ActorCriticNet, Discriminator
# from mocap.mocap import MoCap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOStorage:
    def __init__(self, num_inputs, num_outputs, max_size=64000):
        self.states = torch.zeros(max_size, num_inputs).to(device)
        self.next_states = torch.zeros(max_size, num_inputs).to(device)
        self.actions = torch.zeros(max_size, num_outputs).to(device)
        self.dones = torch.zeros(max_size, 1, dtype=torch.int8).to(device)
        self.log_probs = torch.zeros(max_size).to(device)
        self.rewards = torch.zeros(max_size).to(device)
        self.q_values = torch.zeros(max_size, 1).to(device)
        self.mean_actions = torch.zeros(max_size, num_outputs).to(device)
        self.counter = 0
        self.sample_counter = 0
        self.max_samples = max_size

    def sample(self, batch_size):
        idx = torch.randint(self.counter, (batch_size,), device=device)
        return self.states[idx, :], self.actions[idx, :], self.next_states[idx, :], self.rewards[idx], self.q_values[idx, :], self.log_probs[idx]

    def clear(self):
        self.counter = 0

    def push(self, states, actions, next_states, rewards, q_values, log_probs, size):
        self.states[self.counter:self.counter + size, :] = states.detach().clone()
        self.actions[self.counter:self.counter + size, :] = actions.detach().clone()
        self.next_states[self.counter:self.counter + size, :] = next_states.detach().clone()
        self.rewards[self.counter:self.counter + size] = rewards.detach().clone()
        self.q_values[self.counter:self.counter + size, :] = q_values.detach().clone()
        self.log_probs[self.counter:self.counter + size] = log_probs.detach().clone()
        self.counter += size

    def discriminator_sample(self, batch_size):
        if self.sample_counter == 0 or self.sample_counter == self.max_samples:
            self.permute()
        self.sample_counter %= self.max_samples
        self.sample_counter += batch_size

        return self.states[self.sample_counter - batch_size:self.sample_counter, :], self.next_states[self.sample_counter - batch_size:self.sample_counter, :]

    def critic_sample(self, batch_size):
        if self.sample_counter == 0 or self.sample_counter == self.max_samples:
            self.permute()
        self.sample_counter %= self.max_samples
        self.sample_counter += batch_size
        return self.states[self.sample_counter - batch_size:self.sample_counter, :], self.q_values[self.sample_counter - batch_size:self.sample_counter,:]

    def actor_sample(self, batch_size):
        if self.sample_counter == 0 or self.sample_counter == self.max_samples:
            self.permute()
        self.sample_counter %= self.max_samples
        self.sample_counter += batch_size
        return self.states[self.sample_counter - batch_size:self.sample_counter, :], self.actions[self.sample_counter - batch_size:self.sample_counter, :], self.q_values[self.sample_counter - batch_size:self.sample_counter, :], self.log_probs[self.sample_counter - batch_size:self.sample_counter]

    def permute(self):
        permuted_index = torch.randperm(self.max_samples)
        self.states[:, :] = self.states[permuted_index, :]
        self.actions[:, :] = self.actions[permuted_index, :]
        self.q_values[:, :] = self.q_values[permuted_index, :]
        self.log_probs[:] = self.log_probs[permuted_index]


class RL(object):
    def __init__(self, env, args):

        self.env = env
        self.args = args
        self.num_inputs = int(env.single_observation_space.shape[0])
        self.num_outputs = int(env.single_action_space.shape[0])

        # self.time_step = env.get_attr('time_step')[0]
        self.num_envs = env.num_envs

        self.num_env_steps = self.args.num_steps  # int(args.max_ep_time / self.time_step *args.frame_skip)

        self.Net = ActorCriticNet
        self.model = self.Net(self.num_inputs, self.num_outputs, args.agent_hidden_layers)

        obs_size = env.single_observation_space.shape[0]

        if args.disc_hidden_layers is None:
            args.disc_hidden_layers = [128, 128]

        self.discriminator = Discriminator((obs_size-1) * 2, args.disc_hidden_layers)

        self.model.share_memory()
        self.test_mean = []
        self.test_std = []

        self.noisy_test_mean = []
        self.noisy_test_std = []
        self.lr = 1e-3

        self.clip = .2
        self.test_list = []
        self.noisy_test_list = []

        self.best_score_queue = mp.Queue()
        self.best_score = mp.Value("f", 0)
        self.max_reward = mp.Value("f", 1)

        self.best_validation = 1.0
        self.current_best_validation = 1.0

        self.gpu_model = self.Net(self.num_inputs, self.num_outputs, args.agent_hidden_layers)
        self.gpu_model.to(device)
        self.model_old = self.Net(self.num_inputs, self.num_outputs, args.agent_hidden_layers).to(device)
        self.discriminator.to(device)

        self.base_controller = None
        self.base_policy = None

        self.total_rewards = []
        self.episode_lengths = []

        self.actor_optimizer = optim.AdamW(self.gpu_model.parameters(), lr=1e-4)
        self.critic_optimizer = optim.AdamW(self.gpu_model.parameters(), lr=1e-4)

    def collect_samples_vec(self, num_samples, start_state=None, noise=-2.5, env_index=0, random_seed=1):

        start_state = np.asarray(self.env.get_attr('get_obs'))

        samples = 0
        states = []
        next_states = []
        actions = []
        rewards = []
        q_values = []
        log_probs = []
        dones = []
        terminateds = []

        noise = self.base_noise * self.explore_noise.value
        self.gpu_model.set_noise(noise)

        state = start_state

        state = torch.from_numpy(state).type(torch.cuda.FloatTensor).to(device) if device =='cuda' else torch.from_numpy(state).type(torch.FloatTensor).to(device)
        while samples < num_samples:

            with torch.no_grad():
                action, mean_action = self.gpu_model.sample_actions(state)
                log_prob = self.gpu_model.calculate_prob(state, action, mean_action)

            states.append(state.clone())
            actions.append(action.clone())
            log_probs.append(log_prob.clone())
            next_state, reward, terminated, done, info = self.env.step(action.cpu().numpy())

            unmodified_next_state = next_state.copy()
            unmodified_next_state = torch.from_numpy(unmodified_next_state).type(torch.cuda.FloatTensor).to(device)  # if device =='cuda' else torch.from_numpy(next_state).type(torch.FloatTensor).to(device)

            next_state = next_state.copy()

            if len(info.keys()) != 0:
                next_state[info['_final_observation']] = info['final_observation'][info['_final_observation']][0]

            next_state = torch.from_numpy(next_state).type(torch.cuda.FloatTensor).to(device) if device =='cuda' else torch.from_numpy(next_state).type(torch.FloatTensor).to(device)
            reward = torch.from_numpy(reward).type(torch.cuda.FloatTensor).to(device) if device =='cuda' else torch.from_numpy(reward).type(torch.FloatTensor).to(device)
            terminated = torch.from_numpy(np.array(terminated)).type(torch.cuda.IntTensor).to(device) if device =='cuda' else torch.from_numpy(np.array(terminated)).type(torch.IntTensor).to(device)
            done = torch.from_numpy(np.array(done)).type(torch.cuda.IntTensor).to(device) if device =='cuda' else torch.from_numpy(np.array(done)).type(torch.IntTensor).to(device)

            terminateds.append(terminated.clone())
            dones.append(done.clone())
            next_states.append(next_state.clone())

            if self.args.alg == 'amp':
                reward = self.discriminator.compute_disc_reward(state[:, :-1], next_state[:, :-1]) * .1 # DISC
            rewards.append(reward.clone())

            state = unmodified_next_state.clone()

            samples += 1

        counter = num_samples - 1
        R = self.gpu_model.get_value(next_state)
        while counter >= 0:
            R = self.gpu_model.get_value(next_states[counter]) * dones[counter].unsqueeze(-1) + (1 - dones[counter].unsqueeze(-1)) * R
            R = R * (1 - terminateds[counter].unsqueeze(-1))
            R = 0.99 * R + rewards[counter].unsqueeze(-1)
            q_values.insert(0, R)
            counter -= 1

        for i in range(num_samples):
            self.storage.push(states[i], actions[i], next_states[i], rewards[i], q_values[i], log_probs[i], self.num_envs)

    def sample_expert_motion(self, batch_size):
        states, next_states = self.mocap.sample_expert(batch_size)

        return torch.from_numpy(states).float().to(device), torch.from_numpy(next_states).float().to(device)

    def update_discriminator(self, batch_size, num_epoch):
        self.discriminator.train()
        optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr)
        epoch_loss = 0

        for k in range(num_epoch):
            batch_states, batch_next_states = self.storage.discriminator_sample(batch_size)

            policy_d = self.discriminator.compute_disc(batch_states[:, :-1], batch_next_states[:, :-1]) # # DISC
            policy_loss = (policy_d + torch.ones(policy_d.size(), device=device)) ** 2
            policy_loss = policy_loss.mean()

            batch_expert_states, batch_expert_next_states = self.sample_expert_motion(batch_size)

            expert_d = self.discriminator.compute_disc(batch_expert_states, batch_expert_next_states)
            expert_loss = (expert_d - torch.ones(expert_d.size(), device=device)) ** 2
            expert_loss = expert_loss.mean()

            grad_penalty = self.discriminator.grad_penalty(batch_expert_states, batch_expert_next_states)

            total_loss = policy_loss + expert_loss + 5 * grad_penalty
            epoch_loss += total_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        return epoch_loss/num_epoch

    def update_critic(self, batch_size, num_epoch):
        self.gpu_model.train()
        optimizer = self.critic_optimizer

        storage = self.storage
        gpu_model = self.gpu_model
        epoch_loss = 0

        for k in range(num_epoch):
            batch_states, batch_q_values = storage.critic_sample(batch_size)
            v_pred = gpu_model.get_value(batch_states)

            loss_value = (v_pred - batch_q_values) ** 2
            loss_value = 0.5 * loss_value.mean()
            epoch_loss += loss_value

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        return epoch_loss/num_epoch

    def update_actor(self, batch_size, num_epoch):
        self.gpu_model.train()
        optimizer = self.actor_optimizer

        storage = self.storage
        gpu_model = self.gpu_model
        model_old = self.model_old
        params_clip = self.clip

        epoch_loss = 0

        for k in range(num_epoch):
            batch_states, batch_actions, batch_q_values, batch_log_probs = storage.actor_sample(batch_size)

            batch_q_values = batch_q_values  # / self.max_reward.value

            with torch.no_grad():
                v_pred_old = gpu_model.get_value(batch_states)

            batch_advantages = (batch_q_values - v_pred_old)

            probs, mean_actions = gpu_model.calculate_prob_gpu(batch_states, batch_actions)
            probs_old = batch_log_probs  # model_old.calculate_prob_gpu(batch_states, batch_actions)
            ratio = (probs - (probs_old)).exp()
            ratio = ratio.unsqueeze(1)
            surr1 = ratio * batch_advantages
            surr2 = ratio.clamp(1 - params_clip, 1 + params_clip) * batch_advantages
            loss_clip = -(torch.min(surr1, surr2)).mean()

            total_loss = loss_clip + 0.001 * (mean_actions ** 2).mean()
            epoch_loss += total_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if self.lr > 1e-4:
            self.lr *= 0.99
        else:
            self.lr = 1e-4

        return epoch_loss/num_epoch

    def save_model(self, filename):
        torch.save(self.gpu_model.state_dict(), filename)

    def train(self):
        self.start = time.time()
        self.lr = 1e-3
        self.weight = 10
        self.num_samples = 0
        self.time_passed = 0

        max_samples = self.num_envs * self.num_env_steps #CHANGE
        # print(max_samples)
        self.storage = PPOStorage(self.num_inputs, self.num_outputs, max_size=max_samples)

        self.explore_noise = mp.Value("f", -2.0)  # -2
        self.base_noise = np.ones(self.num_outputs)
        noise = self.base_noise * self.explore_noise.value
        self.model.set_noise(noise)
        self.gpu_model.set_noise(noise)
        self.env.reset()

        for iterations in range(200000):

            print("-" * 50)
            print("iteration: ", iterations)
            iteration_start = time.time()

            while self.storage.counter < max_samples:
                self.collect_samples_vec(self.num_env_steps//2, noise=noise)

            critic_loss = self.update_critic(max_samples // 4, 40)
            actor_loss = self.update_actor(max_samples // 4, 40)

            if self.args.alg == 'amp':
                disc_loss = self.update_discriminator(max_samples // 4, 40)
                if self.args.wandb:
                    wandb.log({"step": iterations, "train/disc loss": disc_loss})

            self.storage.clear()

            if (iterations) % 5 == 0:
                print("reward: ", np.round(np.mean(self.env.return_queue), 3), u"\u00B1", np.round(np.std(self.env.return_queue), 3))
                print("ep len: ", np.round(np.mean(self.env.length_queue), 3), u"\u00B1", np.round(np.std(self.env.length_queue), 3))

                if self.args.wandb:
                    wandb.log({"step": iterations, "eval/reward": np.mean(self.env.return_queue), "eval/ep_len": np.mean(self.env.length_queue)} )

            if self.args.wandb:
                wandb.log({"step": iterations, "train/critic loss": critic_loss, "train/actor loss": actor_loss})

            print("iteration time", np.round(time.time() - iteration_start, 3))
            print()

            if (iterations) % 100 == 0:
                best_model_path = 'results/' + 'models/' + self.args.exp_name
                os.makedirs(best_model_path, exist_ok=True)
                torch.save(self.gpu_model.state_dict(), best_model_path + '/' + self.args.exp_name + "_iter%d.pt" % (iterations))

        self.save_model('results/' + 'models/' + self.args.exp_name + '/' + self.args.exp_name + "_final.pt")






