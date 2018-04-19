from policy_gradient import BasePolicyGradient

import torch
from torch.autograd import Variable
from torch.distributions import Categorical


class Reinforce(BasePolicyGradient):
    def __init__(self, config, policy, optim):
        super(Reinforce, self).__init__(config, policy, optim)

    def select_action(self, state):
        log_prob = self.policy(Variable(state)).view(-1)
        probs = torch.exp(log_prob)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(log_prob[action])
        self.entropies.append(-1 * torch.dot(probs, log_prob))
        return action.data

    def finish_episode(self):
        R = 0
        policy_loss = 0
        rewards = []
        for r in self.rewards[::-1]:
            R = r + self.config.gamma * R
            rewards.insert(0, R)

        rewards = torch.Tensor(rewards)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        for log_prob, entropy, reward in zip(self.saved_log_probs, self.entropies, rewards):
            policy_loss += -log_prob * reward -  0.0001 * entropy

        self.policy_optim.zero_grad()
        policy_loss.backward()

        total_norm = torch.nn.utils.clip_grad_norm(self.policy.parameters(), self.config.clip)

        self.policy_optim.step()
        self.clear()
        self.policy.clear()
        return policy_loss, total_norm
