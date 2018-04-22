import argparse
import numpy as np
import os
import math
import time
import sys

from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.autograd import Variable
from torch.distributions import Categorical

from tensorboardX import SummaryWriter

from lm_env import LanguageModelingEnv
from models.policies import FeedForwardPolicy, RNNPolicy
from models.critics import FeedForwardCritic, RNNCritic
from train.reinforce import Reinforce
from train.actor_critic import ActorCritic
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
parser.add_argument('--episodes', type=int, default=8000000,
                    help='Number of episodes.')
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
parser.add_argument('--mc_baseline', action='store_true',
                    help='Use Monte Carlo baseline.')
parser.add_argument('--param_baseline', action='store_true',
                    help='Use parametric per state baseline.')
parser.add_argument('--algo', type=str, default='ac',
                    help='Algo to be used. Option: [ac, reinforce]')
parser.add_argument('--parallel', action='store_true',
                    help='Should we run training in parallel.')
parser.add_argument('--num_processes', type=int, default=8,
                    help='Number of processes.')
parser.add_argument('--validate_freq', type=int, default=1000,
                    help='Validate Frequency.')
parser.add_argument('--init_epsilon', type=float, default=0.8,
                    help='Initial epsilon value.')
parser.add_argument('--min_epsilon', type=float, default=0.01,
                    help='Min epsilon value.')
parser.add_argument('--incl_unk_reward', action='store_true',
                    help='Include <unk> word in reward.')
parser.add_argument('--tensorboard', action='store_true',
                    help='Log to tensorboard')
parser.add_argument('--log_dir', type=str, default='./log/',
                    help='Log directory for tensorboard')
parser.add_argument('--print_sentence', action='store_true',
                    help='Print Sentence.')

args = parser.parse_args()

corpus = Corpus(args.data)
vocab_size = len(corpus.dictionary)

env = LanguageModelingEnv(args, corpus.train, bos=torch.LongTensor([0]), eos=torch.LongTensor([1]))
env.seed(args.seed)

if args.algo == 'reinforce':
    policy = RNNPolicy(args, vocab_size)

    if args.parallel:
        policy.share_memory()

    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    algo = Reinforce(args, policy, optimizer)

elif args.algo == 'ac':
    policy = RNNPolicy(args, vocab_size)
    critic = RNNCritic(args, vocab_size)

    if args.parallel:
        policy.share_memory()
        critic.share_memory()

    actor_optim = optim.Adam(policy.parameters(), lr=args.lr)
    critic_optim = optim.Adam(critic.parameters(), lr=args.lr)

    algo = ActorCritic(args, policy, actor_optim, critic, critic_optim)

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

def train(rank, env, valid, args, algo, episodes, seed, writer=None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    start_time = time.time()
    algo.train()
    for episode in range(episodes):
        state = env.reset()
        cuml_reward = 0.0
        for i in range(args.max_len):
            action = algo.select_action(state)
            state, reward, done, predicted_ngrams = env.step(action)
            algo.rewards.append(reward)
            cuml_reward += reward

            if done:
                break

        loss, total_norm = algo.finish_episode()

        predicted_ngrams = [' '.join([corpus.dictionary.idx2word[j] for j in ngram]) for ngram in predicted_ngrams]

        if writer is not None and args.tensorboard:
            writer.add_scalar('data/sentence_len', i, episode)
            writer.add_scalar('data/reward', cuml_reward, episode)
            writer.add_text('data/ngrams', ', '.join(predicted_ngrams), episode)

        print("Rank: %d | Episode: %d | Sent len: %d | Sent Reward: %.3f | Loss: %.3f | ngrams: %s "
                % (rank, episode, i, cuml_reward, loss, ', '.join(predicted_ngrams)))

        if args.print_sentence and (episode + 1) % args.log_interval == 0:
            sentence =  ' '.join([corpus.dictionary.idx2word[j] for j in state])
            print sentence
            writer.add_text('data/sentence', sentence, episode)

        sys.stdout.flush()

        if (episode + 1) % args.validate_freq == 0:
            val_loss = evaluate(valid)
            print('-' * 89)
            print('| end of episode {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'
                        .format(episode, (time.time() - start_time), val_loss, math.exp(min(100, val_loss))))
            print('-' * 89)

            writer.add_scalar('data/valid_ppl', math.exp(min(100, val_loss), episode))

writer = SummaryWriter(args.log_dir)

if False:
# Run on train data.
    valid_loss = evaluate(corpus.valid)
    print('=' * 89)
    print('| Before training | valid loss {:5.2f} | valid ppl {:8.2f}'.format(valid_loss, math.exp(valid_loss)))
    print('=' * 89)

if args.parallel:
    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(rank, env, corpus.valid, args, algo, args.episodes, np.random.randint(0, 10000000)))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
else:
    train(0, env, corpus.valid, args, algo, args.episodes, args.seed, writer)

# Run on test data.
test_loss = evaluate(corpus.test)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
print('=' * 89)

writer.close()
