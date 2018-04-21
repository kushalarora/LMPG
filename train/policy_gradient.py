from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np
import torch

class BasePolicyGradient(object):
    def __init__(self, config, policy, policy_optim, critic=None, critic_optim=None, behavior_policy=None):
        self.config = config

        self.init_epsilon = config.init_epsilon
        self.epsilon = config.init_epsilon
        self.min_epsilon = config.min_epsilon
        self.current_episode = 0

        self.policy = policy
        self.policy_optim = policy_optim

        self.critic = critic
        self.critic_optim = critic_optim

        self.behavior_policy = behavior_policy \
                                if behavior_policy is not None else \
                                    self.policy

        self.saved_log_probs = []
        self.rewards = []
        self.entropies = []

    def select_action(self, state, test=False):
        raise NotImplementedError()

    def finish_episode(self):
        raise NotImplementedError()

    def update_epsilon(self):
        self.epsilon = self.init_epsilon - float(self.current_episode) / self.config.episodes * (self.init_epsilon - self.min_epsilon)
        self.epsilon = max(self.epsilon, self.min_epsilon)
        self.current_episode += 1

    def sample(self, action_values, deterministic=False):
        if deterministic:
            return np.argmax(action_values)
        if np.random.rand() < self.epsilon:
            random_action = Variable(torch.LongTensor([np.random.randint(0, len(action_values))]))
            return random_action

        m = Categorical(action_values)
        return m.sample()

    def train(self):
        self.policy.train()

        if self.critic:
            self.critic.train()

    def clear(self):
        del self.saved_log_probs[:]
        del self.entropies[:]
        del self.rewards[:]
        self.saved_log_probs = []
        self.rewards = []
        self.entropies = []

        self.update_epsilon()
