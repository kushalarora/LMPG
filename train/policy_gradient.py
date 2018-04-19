class BasePolicyGradient(object):
    def __init__(self, config, policy, policy_optim, critic=None, critic_optim=None, behavior_policy=None):
        self.config = config
        self.policy = policy
        self.policy_optim = policy_optim

        self.critic = critic
        self.critic_optim = critic_optim

        self.behavior_policy = behavior_policy

        self.saved_log_probs = []
        self.rewards = []
        self.entropies = []

    def select_action(self, state):
        raise NotImplementedError()

    def finish_episode(self):
        raise NotImplementedError()

    def clear(self):
        del self.saved_log_probs[:]
        del self.entropies[:]
        del self.rewards[:]
        self.saved_log_probs = []
        self.rewards = []
        self.entropies = []
