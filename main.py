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
from models.policies import FeedForwardPolicy, RNNPolicy, NgramPolicy
from models.critics import FeedForwardCritic, RNNCritic
from train.reinforce import Reinforce
from train.actor_critic import ActorCritic
from train.ppo import PPO
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
parser.add_argument('--lm_path', type=str, default='data/penn/train.arpa',
                    help='LM path.')
parser.add_argument('--use_behav_pol', action='store_true',
                    help='Use Behavioral policy.')
parser.add_argument('--gpuid', type=int, default=0,
                    help='GPU ID.')
parser.add_argument('--batch_size', type=int, default=1000,
                    help='Batch of trajectories before optimizing')
parser.add_argument('--log_dir_prefix', type=str, default='log_dir_prefix',
                    help='Prefix for log directory.')
parser.add_argument('--sparse_rewards', action='store_true',
                    help='Sparse Rewards.')
args = parser.parse_args()

corpus = Corpus(args.data)
vocab_size = len(corpus.dictionary)

bos, eos = (torch.LongTensor([0]), torch.LongTensor([1]))
if args.cuda:
    bos, eos = (bos.cuda(args.gpuid), eos.cuda(args.gpuid))

env = LanguageModelingEnv(args, corpus.train, bos=bos, eos=eos)
env.seed(args.seed)

behavior_policy = None
if args.use_behav_pol:
    behavior_policy = NgramPolicy(args, corpus.dictionary)

if args.algo == 'reinforce':
    policy = RNNPolicy(args, vocab_size)

    if args.cuda:
        policy = policy.cuda(args.gpuid)

    if args.parallel:
        policy.share_memory()

    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    algo = Reinforce(args, policy, optimizer, behavior_policy=behavior_policy)

elif args.algo == 'ac':
    policy = RNNPolicy(args, vocab_size)
    critic = RNNCritic(args, vocab_size)

    if args.cuda:
        policy = policy.cuda(args.gpuid)
        critic = critic.cuda(args.gpuid)

    if args.parallel:
        policy.share_memory()
        critic.share_memory()

    actor_optim = optim.Adam(policy.parameters(), lr=args.lr)
    critic_optim = optim.Adam(critic.parameters(), lr=args.lr)

    algo = ActorCritic(args, policy, actor_optim, critic, critic_optim, behavior_policy=behavior_policy)
elif args.algo == 'ppo':
    policy = RNNPolicy(args, vocab_size)
    critic = RNNCritic(args, vocab_size)

    if args.cuda:
        policy = policy.cuda(args.gpuid)
        critic = critic.cuda(args.gpuid)

    actor_optim = optim.Adam(policy.parameters(), lr=args.lr)
    critic_optim = optim.Adam(critic.parameters(), lr=args.lr)

    algo = PPO(args, policy, actor_optim, critic, critic_optim, behavior_policy=behavior_policy)

criterion = nn.CrossEntropyLoss()

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda(args.gpuid)
    return data

train_data = batchify(corpus.train, 32)
valid_data = batchify(corpus.valid, 32)
test_data = batchify(corpus.test, 32)

def cudafy(source, args):
    if args.cuda:
        source = source.cuda(args.gpuid)
    return source

def get_batch(source, i, evaluation=False):
    seq_len = min(args.max_len, len(source) - 1 - i)
    data = Variable(cudafy(source[i:i+seq_len], args), volatile=evaluation)
    target = Variable(cudafy(source[i+1:i+1+seq_len].view(-1), args))
    return data, target

def evaluate(data_source, policy):
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

def train(rank, env, train_data, valid_data, args, algo, episodes, seed, writer=None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    start_time = time.time()
    algo.train()
    for episode in range(episodes):
        state = env.reset()

        if args.cuda:
            state = state.cuda(args.gpuid)
        cuml_reward = 0.0
        for i in range(args.max_len):
            action = algo.select_action(state)
            state, reward, done, predicted_ngrams = env.step(action)
            algo.rewards.append(reward)
            cuml_reward += reward

            if done:
                break

        loss = algo.finish_episode()

        if episode % args.batch_size == 0:
            algo.update_params()

        predicted_ngrams = [' '.join([corpus.dictionary.idx2word[j] for j in ngram]) for ngram in predicted_ngrams]

        if (writer is not None) and args.tensorboard:
            writer.add_scalar('data/sentence_len', i, episode)
            writer.add_scalar('data/reward', cuml_reward, episode)
            writer.add_text('data/ngrams', ', '.join(predicted_ngrams), episode)

        print("Rank: %d | Episode: %d | Sent len: %d | Sent Reward: %.3f | epsilon: %.3f | ngrams: %s "
                % (rank, episode, i, cuml_reward, algo.epsilon, ', '.join(predicted_ngrams)))

        if args.print_sentence and (episode + 1) % args.log_interval == 0:
            sentence =  ' '.join([corpus.dictionary.idx2word[j] for j in state])
            print sentence
            writer.add_text('data/sentence', sentence, episode)

        sys.stdout.flush()

        if (episode + 1) % args.validate_freq == 0:
            train_loss = evaluate(train_data, policy)
            val_loss = evaluate(valid_data, policy)
            print('-' * 89)
            print('| end of episode {:3d} | time: {:5.2f}s | train loss {:5.2f} | valid loss {:8.2f}'
                        .format(episode, (time.time() - start_time), train_loss, valid_loss))
            print('-' * 89)

            writer.add_scalar('data/train_entropy', train_loss, episode)
            writer.add_scalar('data/valid_entropy', val_loss, episode)

subfolder = os.path.join(args.log_dir, "%s_%s_%.4f_%s" % (args.log_dir_prefix, args.algo, args.lr, time.strftime("%Y_%m_%d_%H_%M_%S")))
os.mkdir(subfolder)
writer = SummaryWriter(subfolder)

if True:
# Run on train data.
    train_loss = evaluate(train_data, policy)
    valid_loss = evaluate(valid_data, policy)
    if args.use_behav_pol:
        train_loss = evaluate(corpus.train, behavior_policy)
        valid_loss = evaluate(corpus.valid, behavior_policy)
        pass
    print('=' * 89)
    print('| Before training | train loss {:5.2f} | valid loss {:8.2f}'.format(valid_loss, math.exp(valid_loss)))
    print('=' * 89)

if args.parallel:
    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(rank, env, train_data, valid_data, args, algo, args.episodes, np.random.randint(0, 10000000)))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
else:
    train(0, env, train_data, valid_data, args, algo, args.episodes, args.seed, writer)

# Run on test data.
test_loss = evaluate(test_data, policy)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
print('=' * 89)

writer.close()
