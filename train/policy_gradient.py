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
        self.kl_terms = []

    def select_action(self, state, test=False):
        raise NotImplementedError()

    def finish_episode(self):
        raise NotImplementedError()

    def update_params(self):
        raise NotImplementedError()

    def update_epsilon(self):
        self.epsilon = self.init_epsilon - float(self.current_episode) / self.config.episodes * (self.init_epsilon - self.min_epsilon)
        self.epsilon = max(self.epsilon, self.min_epsilon)
        self.current_episode += 1

    def sample(self, state, deterministic=False):
        state = Variable(state)

        log_prob_policy = self.policy(state).view(-1)
        if np.random.rand() < self.epsilon:
            if self.config.use_behav_pol:
                log_prob = self.behavior_policy(state).view(-1)
                action = Categorical(torch.exp(log_prob)).sample()
                ratio = torch.exp(log_prob_policy[action])/torch.exp(log_prob[action])
            else:
                action = Variable(state.data.new([np.random.randint(0, len(log_prob_policy))]))
                ratio = torch.exp(log_prob_policy[action])/(1.0/len(log_prob_policy))
        else:
            if deterministic:
                action = np.argmax(log_prob_policy)
            else:
                action = Categorical(torch.exp(log_prob_policy)).sample()
            ratio = 1

        return action, ratio, log_prob_policy

    def train(self):
        self.policy.train()

        if self.critic:
            self.critic.train()

    def clear(self):
        del self.saved_log_probs[:]
        del self.entropies[:]
        del self.rewards[:]
        del self.kl_terms[:]

        self.saved_log_probs = []
        self.rewards = []
        self.entropies = []
        self.kl_terms = []

        self.update_epsilon()
