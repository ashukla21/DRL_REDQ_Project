import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import logging

from ..llm_interface import LLMInterface
from ..user_config import LLM_CONFIG
from redq.algos.core import TanhGaussianPolicy, Mlp, soft_update_model1_with_model2, ReplayBuffer, mbpo_target_entropy_dict


def get_probabilistic_num_min(num_mins):
    floored_num_mins = np.floor(num_mins)
    if num_mins - floored_num_mins > 0.001:
        prob_for_higher_value = num_mins - floored_num_mins
        if np.random.uniform(0, 1) < prob_for_higher_value:
            return int(floored_num_mins + 1)
        else:
            return int(floored_num_mins)
    else:
        return int(num_mins)


class REDQSACAgent(object):
    def __init__(self, env_name, obs_dim, act_dim, act_limit, device,
                 hidden_sizes=(256, 256), replay_size=int(1e6), batch_size=256,
                 lr=3e-4, gamma=0.99, polyak=0.995,
                 alpha=0.2, auto_alpha=True, target_entropy='mbpo',
                 start_steps=5000, delay_update_steps='auto',
                 utd_ratio=20, num_Q=10, num_min=2, q_target_mode='min',
                 policy_update_delay=20):
        self.policy_net = TanhGaussianPolicy(obs_dim, act_dim, hidden_sizes, action_limit=act_limit).to(device)
        self.q_net_list, self.q_target_net_list = [], []
        for _ in range(num_Q):
            q = Mlp(obs_dim + act_dim, 1, hidden_sizes).to(device)
            q_targ = Mlp(obs_dim + act_dim, 1, hidden_sizes).to(device)
            q_targ.load_state_dict(q.state_dict())
            self.q_net_list.append(q)
            self.q_target_net_list.append(q_targ)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q_optimizer_list = [optim.Adam(q.parameters(), lr=lr) for q in self.q_net_list]

        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -act_dim if target_entropy == 'auto' else mbpo_target_entropy_dict[env_name]
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.cpu().exp().item()
        else:
            self.alpha = alpha
            self.target_entropy = self.log_alpha = self.alpha_optim = None

        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        self.mse_criterion = nn.MSELoss()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.lr = lr
        self.hidden_sizes = hidden_sizes
        self.gamma = gamma
        self.polyak = polyak
        self.replay_size = replay_size
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_min = num_min
        self.num_Q = num_Q
        self.utd_ratio = utd_ratio
        self.start_steps = start_steps
        self.delay_update_steps = start_steps if delay_update_steps == 'auto' else delay_update_steps
        self.q_target_mode = q_target_mode
        self.policy_update_delay = policy_update_delay
        self.device = device

        self.total_steps = 0
        self.llm_interface = LLMInterface()

    def __get_current_num_data(self):
        return self.replay_buffer.size

    def get_exploration_action(self, obs, env):
        with torch.no_grad():
            if self.__get_current_num_data() > self.start_steps:
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
                exploration_noise_scale = None

                if LLM_CONFIG.get('use_llm_exploration', False) and \
                   self.llm_interface.client and \
                   self.total_steps % LLM_CONFIG.get('call_frequency', 100) == 0:
                    exploration_noise_scale = self.llm_interface.get_exploration_noise_scale(obs)
                    if exploration_noise_scale is not None:
                        logging.info(f"Step {self.total_steps}: LLM suggested noise scale: {exploration_noise_scale}")

                action_tensor = self.policy_net.forward(obs_tensor, deterministic=False, return_log_prob=False)[0]

                if exploration_noise_scale is not None:
                    noise = torch.normal(mean=0.0, std=exploration_noise_scale, size=action_tensor.shape).to(self.device)
                    action_tensor = action_tensor + noise
                    action_tensor = torch.clamp(action_tensor, -self.act_limit, self.act_limit)

                action = action_tensor.cpu().numpy().reshape(-1)
            else:
                action = env.action_space.sample()
        return action

    def get_test_action(self, obs):
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
            action_tensor = self.policy_net.forward(obs_tensor, deterministic=True, return_log_prob=False)[0]
            return action_tensor.cpu().numpy().reshape(-1)

    def get_action_and_logprob_for_bias_evaluation(self, obs):
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
            action_tensor, _, _, log_prob_a_tilda, _, _ = self.policy_net.forward(obs_tensor, deterministic=False, return_log_prob=True)
            action = action_tensor.cpu().numpy().reshape(-1)
        return action, log_prob_a_tilda

    def get_ave_q_prediction_for_bias_evaluation(self, obs_tensor, acts_tensor):
        q_vals = [q(torch.cat([obs_tensor, acts_tensor], 1)) for q in self.q_net_list]
        return torch.mean(torch.cat(q_vals, dim=1), dim=1)

    def store_data(self, o, a, r, o2, d):
        self.replay_buffer.store(o, a, r, o2, d)
        self.total_steps += 1

    def sample_data(self, batch_size):
        batch = self.replay_buffer.sample_batch(batch_size)
        return (Tensor(batch['obs1']).to(self.device),
                Tensor(batch['obs2']).to(self.device),
                Tensor(batch['acts']).to(self.device),
                Tensor(batch['rews']).unsqueeze(1).to(self.device),
                Tensor(batch['done']).unsqueeze(1).to(self.device))

    def get_redq_q_target_no_grad(self, obs_next_tensor, rews_tensor, done_tensor):
        num_mins_to_use = get_probabilistic_num_min(self.num_min)
        sample_idxs = np.random.choice(self.num_Q, num_mins_to_use, replace=False)

        with torch.no_grad():
            a_tilda_next, _, _, log_prob_a_tilda_next, _, _ = self.policy_net.forward(obs_next_tensor)

            q_vals = [self.q_target_net_list[i](torch.cat([obs_next_tensor, a_tilda_next], 1)) for i in sample_idxs]
            q_vals_cat = torch.cat(q_vals, dim=1)

            if self.q_target_mode == 'min':
                min_q, _ = torch.min(q_vals_cat, dim=1, keepdim=True)
                q_target = min_q - self.alpha * log_prob_a_tilda_next
            elif self.q_target_mode == 'ave':
                mean_q = q_vals_cat.mean(dim=1, keepdim=True)
                q_target = mean_q - self.alpha * log_prob_a_tilda_next
            elif self.q_target_mode == 'rem':
                rem_weight = Tensor(np.random.uniform(0, 1, q_vals_cat.shape)).to(self.device)
                rem_weight /= rem_weight.sum(dim=1, keepdim=True)
                rem_q = (q_vals_cat * rem_weight).sum(dim=1, keepdim=True)
                q_target = rem_q - self.alpha * log_prob_a_tilda_next
            else:
                raise ValueError(f"Unknown Q target mode: {self.q_target_mode}")

            y_q = rews_tensor + self.gamma * (1 - done_tensor) * q_target

        return y_q, sample_idxs