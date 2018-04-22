import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

from pysrilm.srilm import LM

class LanguageBasePolicy(nn.Module):
    def __init__(self, config, vocab_size):
        super(LanguageBasePolicy, self).__init__()
        self.config = config
        self.encoder = nn.Embedding(vocab_size, self.config.embedding_size)
        self.decoder = nn.Linear(self.config.hidden_size, vocab_size)

        if self.config.tie_weights:
            assert self.config.embedding_size == self.config.hidden_size, \
                    "Hidden layer size ({}) != Embedding layer size ({})" \
                        .format(self.config.embedding_size, self.config.hidden_size)
            self.encoder.weight = self.decoder.weight

	self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, state):
	raise NotImplementedError()

    def init_hidden_state(self):
	raise NotImplementedError()

    def clear(self):
        self.init_hidden_state()

class FeedForwardPolicy(LanguageBasePolicy):
    def __init__(self, config, vocab_size):
   	LanguageBasePolicy.__init__(self, config, vocab_size)
        self.hidden_layer = nn.Linear(self.config.context_size * self.config.embedding_size, self.config.hidden_size)

    def forward(self, state):
	context = state[-self.config.context_size:]
        embeds = self.encoder(context).view((1, -1))
        out = F.relu(self.hidden_layer(embeds))
        out = self.decoder(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def init_hidden_state(self):
        pass

class RNNPolicy(LanguageBasePolicy):
    def __init__(self, config, vocab_size):
   	LanguageBasePolicy.__init__(self, config, vocab_size)
        self.hidden_layer = nn.RNNCell(self.config.embedding_size, self.config.hidden_size, nonlinearity=self.config.nonlinearity)
	self.init_hidden_state()

    def forward(self, state):
        embeds = self.encoder(state[-1]).view((1, -1))
        self.hidden_state = self.hidden_layer(embeds, self.hidden_state)
        out = self.decoder(self.hidden_state)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def init_hidden_state(self):
        weight = next(self.parameters()).data
	self.hidden_state = Variable(weight.new(self.config.hidden_size).zero_())


class LSTMPolicy(LanguageBasePolicy):
    def __init__(self, config, vocab_size):
   	LanguageBasePolicy.__init__(self, config, vocab_size)
        self.hidden_layer = nn.LSTMCell(self.config.embedding_size, self.config.hidden_size)
	self.init_hidden_state()

    def forward(self, state):
        embeds = self.encoder(state[-1]).view((1, -1))
        self.hidden_state = self.hidden_layer(embeds, self.hidden_state)
        out = self.linear2(self.hidden_state[0])
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def init_hidden_state(self):
        weight = next(self.parameters()).data
        self.hidden_state = (Variable(weight.new(self.config.hidden_size).zero_()),
                                Variable(weight.new(self.config.hidden_size).zero_()))

class NgramPolicy(LanguageBasePolicy):
    def __init__(self, config, dictionary):
        LanguageBasePolicy.__init__(self, config, len(dictionary))
        self.lm = LM(self.config.lm_path)
        self.dictionary = dictionary

    def forward(self, state):
        log_probs = []
        state = [self.dictionary.idx2word[idx] for idx in reversed(state.tolist())]
        for w in self.dictionary.idx2word:
            log_probs.append(self.lm.logprob_strings(w, state) + 1e-10)
        return torch.Tensor(log_probs)

    def init_hidden_state(self):
        pass
