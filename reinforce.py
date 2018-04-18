import argparse
import numpy as np
import os

from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from lm_env import LanguageModelingEnv
from policies import FeedForwardPolicy, RNNPolicy

from data import Corpus

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--ngrams', type=int, default=4,
                    help='Number of ngrams to consider for Reward.')
parser.add_argument('--hidden_size', type=int, default=200,
                    help='Number of ngrams to consider for Reward.')
parser.add_argument('--embedding_size', type=int, default=200,
                    help='Number of ngrams to consider for Reward.')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='initial learning rate')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--epi_per_epoch', type=int, default=10000,
                    help='upper epoch limit')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tie_weights', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--context_size', type=int, default=4,
                    help='Number of ngrams to consider for Reward.')
parser.add_argument('--nonlinearity', type=str, default='relu',
                    help='Non linearity for rnn.')
parser.add_argument('--max_len', type=int, default=40,
                    help='Maximum number of words in a sentence.')
args = parser.parse_args()

corpus = Corpus(args.data)
vocab_size = len(corpus.dictionary)

env = LanguageModelingEnv(args, corpus.train, bos=torch.LongTensor([0]), eos=torch.LongTensor([1]))
env.seed(args.seed)
torch.manual_seed(args.seed)


policy = RNNPolicy(args, vocab_size)

optimizer = optim.Adam(policy.parameters(), lr=args.lr)


def select_action(state):
    log_prob = policy(Variable(state)).view(-1)
    probs = torch.exp(log_prob)
    try:
        m = Categorical(probs)
        action = m.sample()
        policy.saved_log_probs.append(log_prob[action])
        policy.entropies.append(-1 * torch.dot(probs, log_prob))
    except:
        import pdb;pdb.set_trace()
    return action.data


def finish_episode():
    R = 0
    policy_loss = 0
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    # rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for log_prob, entropy, reward in zip(policy.saved_log_probs, policy.entropies, rewards):
        policy_loss += -log_prob * reward + 0.01 * entropy
    optimizer.zero_grad()
    policy_loss.backward()

    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    total_norm = torch.nn.utils.clip_grad_norm(policy.parameters(), args.clip)
    if np.isnan(total_norm):
        import pdb;pdb.set_trace()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    policy.clear()

    return total_norm

def main():
    for epoch in range(args.epochs):
        for episode in range(args.epi_per_epoch):
            state = env.reset()
            cuml_reward = 0.0
            for i in range(args.max_len):
                action = select_action(state)
                state, reward, done, predicted_ngrams = env.step(action)
                policy.rewards.append(reward)
                cuml_reward += reward

                if done:
                    break

            total_norm = finish_episode()

            print("Episode: %d | Sent len: %d | Sent Reward: %.3f | Norm: %.3f | ngrams: %s"
                    % (episode, i, cuml_reward, total_norm,
                        ', '.join([' '.join([corpus.dictionary.idx2word[j] for j in ngram]) for ngram in predicted_ngrams])))

            if False and (episode + 1) % args.log_interval == 0:
                print ' '.join([corpus.dictionary.idx2word[j] for j in state])
if __name__ == '__main__':
    main()
