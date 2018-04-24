from policy_gradient import BasePolicyGradient

import numpy as np
import torch
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.nn.functional as F


class ActorCritic(BasePolicyGradient):
    def __init__(self, config, actor, actor_optim, critic, critic_optim, behavior_policy=None):
        super(ActorCritic, self).__init__(config, actor, actor_optim, critic, critic_optim, behavior_policy=behavior_policy)
        self.values = []

        self.batch_policy_loss = 0
        self.batch_value_loss = 0

    def select_action(self, state):
        action, ratio, log_prob_behav = self.sample(state)
        log_prob = self.policy(Variable(state)).view(-1)
        value = self.critic(Variable(state)).view(-1)

        probs = torch.exp(log_prob)

        self.saved_log_probs.append(ratio * log_prob[action])
        self.values.append(value)
        self.entropies.append(-1 * torch.dot(probs, log_prob))

        return action.data

    def finish_episode(self):
        R = 0
        policy_loss = 0
        value_loss = 0
        rewards = []

        for r in self.rewards[::-1]:
            R = r + self.config.gamma * R
            rewards.insert(0, R)
        rewards = torch.Tensor(rewards)

        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

        for log_prob, value, entropy, r in zip(self.saved_log_probs, self.values, self.entropies,  rewards):
            reward = r - value.data[0]
            policy_loss += -log_prob * reward - 0.0001 * entropy
            r = torch.Tensor([r])
            if self.config.cuda:
                r = r.cuda(self.config.gpuid)
            value_loss += F.smooth_l1_loss(value, Variable(r))

        self.batch_policy_loss += policy_loss
        self.batch_value_loss += value_loss

        self.clear()
        self.policy.clear()
        self.critic.clear()
        del self.values[:]
        self.values = []

        return policy_loss + value_loss

    def update_params(self):
        self.policy_optim.zero_grad()
        self.critic_optim.zero_grad()

        self.batch_policy_loss.backward()
        self.batch_value_loss.backward()

        total_policy_norm = torch.nn.utils.clip_grad_norm(self.policy.parameters(), self.config.clip)
        total_value_norm = torch.nn.utils.clip_grad_norm(self.critic.parameters(), self.config.clip)

        self.policy_optim.step()
        self.critic_optim.step()

        self.batch_policy_loss = 0
        self.batch_value_loss = 0

