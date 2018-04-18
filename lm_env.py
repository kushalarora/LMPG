from __future__ import division
from collections import defaultdict

import gym
import math
import torch

class LanguageModelingEnv(gym.Env):
    def __init__(self, config, train, bos, eos):
        self.config = config
        self.corpus_ngrams = defaultdict(int)
        _update_ngrams_count(train, config.ngrams, self.corpus_ngrams)

        self.last_state = None
        self.last_score = 0

        self.BOS = bos 
        self.EOS = eos 

    def step(self, action):
        next_state = torch.cat([self.last_state, action])
        reward = self._reward(self.last_state, action) 
        self.last_state = next_state
        return next_state, reward, (action == self.EOS).sum(), "" 

    def reset(self):
        self.last_state = self.BOS 
        self.last_score = 0
        return self.BOS

    def _reward(self, state, action):
       current_score = self._score_sentence(torch.cat([state, action]))
       reward = current_score - self.last_score 
       self.last_score = current_score
       return reward 

    def _score_sentence(self, pred):
        score = 0.0
        # Init ngrams count for pred to 0.
        count_pred = defaultdict(int)

        len_pred = len(pred)
        for i in range(len_pred):
            for n in range(1, self.config.ngrams + 1):
                if i - n + 1 < 0:
                    continue

                # n-gram is from i - n to i.
                ngram = tuple(pred[(i - n + 1) : (i + 1)])

                # Update n-gram count.
                count_pred[ngram] += 1
        
        for ngram, val in count_pred.iteritems():
            if ngram in self.corpus_ngrams: 
                # Penality for repitition. * Penality for predicting very common ngram.
                score += (1/count_pred[ngram]) * (1/self.corpus_ngrams[ngram])  
        return score

def _update_ngrams_count(train, ngrams, count):
    length = len(train)
    for n in range(1, ngrams + 1):
        for i in range(length - n + 1):
            ngram = tuple(train[i : (i + n)])
            count[ngram] += 1