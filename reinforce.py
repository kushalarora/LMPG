import argparse
import numpy as np
import os
import math
import time

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

criterion = nn.CrossEntropyLoss()
def get_batch(source, i, evaluation=False):
    seq_len = min(args.max_len, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

def evaluate(data_source):
    # turn on evaluation mode which disables dropout.
    policy.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    for i in range(0, data_source.size(0) - 1, args.max_len):
        data, targets = get_batch(data_source, i, evaluation=True)
        log_probs = []
        for j in range(1, len(data) + 1):
            log_probs.append(policy(data[:j]).view(-1))
        log_probs = torch.stack(log_probs).view(-1, ntokens)
        total_loss += len(data) * criterion(log_probs, targets).data
        policy.clear()
    return total_loss[0] / len(data_source)

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
        policy_loss += -log_prob * reward + 0.0001 * entropy
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

# Run on train data.
valid_loss = evaluate(corpus.valid)
print('=' * 89)
print('| Before training | valid loss {:5.2f} | valid ppl {:8.2f}'.format(valid_loss, math.exp(valid_loss)))
print('=' * 89)

for epoch in range(args.epochs):
    epoch_start_time = time.time()
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

        print("| Episode: %d | Sent len: %d | Sent Reward: %.3f | Norm: %.3f | ngrams: %s "
                % (episode, i, cuml_reward, total_norm,
                    ', '.join([' '.join([corpus.dictionary.idx2word[j] for j in ngram]) for ngram in predicted_ngrams])))

        if False and (episode + 1) % args.log_interval == 0:
            print ' '.join([corpus.dictionary.idx2word[j] for j in state])

    val_loss = evaluate(corpus.valid)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'
                .format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
    print('-' * 89)

# Run on test data.
test_loss = evaluate(corpus.test, policy)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
print('=' * 89)
