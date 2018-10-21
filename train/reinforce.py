from policy_gradient import BasePolicyGradient

import numpy as np
import torch
from torch.autograd import Variable


class Reinforce(BasePolicyGradient):
    def __init__(self, config, policy, optim, behavior_policy=None):
        super(Reinforce, self).__init__(config, policy, optim, behavior_policy=behavior_policy)
        self.batch_policy_loss = 0

    def select_action(self, state):
        action, ratio, kl_term = self.sample(state)
        log_prob = self.policy(Variable(state)).view(-1)
        probs = torch.exp(log_prob)

        self.saved_log_probs.append(ratio * log_prob[action])
        self.entropies.append(-1 * torch.dot(probs, log_prob))
        self.kl_terms.append(kl_term)

        return action.data

    def update_params(self):
        self.policy_optim.zero_grad()

        self.batch_policy_loss.backward()

        total_policy_norm = torch.nn.utils.clip_grad_norm(self.policy.parameters(), self.config.clip)

        self.policy_optim.step()

        self.batch_policy_loss = 0

    def finish_episode(self):
        R = 0
        policy_loss = 0
        rewards = []
        for r in self.rewards[::-1]:
            R = r + self.config.gamma * R
            rewards.insert(0, R)

        rewards = torch.Tensor(rewards)

        for log_prob, entropy, reward in zip(self.saved_log_probs, self.entropies, rewards):
            policy_loss += -log_prob * reward


        self.batch_policy_loss += policy_loss
        self.batch_policy_loss += self.config.kl_term * sum(self.kl_terms)

        self.clear()
        self.policy.clear()
        return policy_loss
