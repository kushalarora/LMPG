import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

class LanguageBaseCritic(nn.Module):
    def __init__(self, config, vocab_size):
        super(LanguageBaseCritic, self).__init__()
        self.config = config
        self.encoder = nn.Embedding(vocab_size, self.config.embedding_size)
        self.decoder = nn.Linear(self.config.hidden_size, 1)

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

class FeedForwardCritic(LanguageBaseCritic):
    def __init__(self, config, vocab_size):
   	LanguageBaseCritic.__init__(self, config, vocab_size)
        self.hidden_layer = nn.Linear(self.config.context_size * self.config.embedding_size, self.config.hidden_size)

    def forward(self, state):
	context = state[-self.config.context_size:]
        embeds = self.encoder(context).view((1, -1))
        out = F.relu(self.hidden_layer(embeds))
        out = self.decoder(out)
        return out

    def init_hidden_state(self):
        pass

class RNNCritic(LanguageBaseCritic):
    def __init__(self, config, vocab_size):
   	LanguageBaseCritic.__init__(self, config, vocab_size)
        self.hidden_layer = nn.RNNCell(self.config.embedding_size, self.config.hidden_size, nonlinearity=self.config.nonlinearity)
	self.init_hidden_state()

    def forward(self, state):
        embeds = self.encoder(state[-1]).view((1, -1))
        self.hidden_state = self.hidden_layer(embeds, self.hidden_state)
        out = self.decoder(self.hidden_state)
        return out

    def init_hidden_state(self):
        weight = next(self.parameters()).data
	self.hidden_state = Variable(weight.new(self.config.hidden_size).zero_())

        if self.config.cuda:
            self.hidden_state = self.hidden_state.cuda(self.config.gpuid)

class LSTMCritic(LanguageBaseCritic):
    def __init__(self, config, vocab_size):
   	LanguageBaseCritic.__init__(self, config, vocab_size)
        self.hidden_layer = nn.LSTMCell(self.config.embedding_size, self.config.hidden_size)
	self.init_hidden_state()

    def forward(self, state):
        embeds = self.encoder(state[-1]).view((1, -1))
        self.hidden_state = self.hidden_layer(embeds, self.hidden_state)
        out = self.linear2(self.hidden_state[0])
        return out

    def init_hidden_state(self):
        weight = next(self.parameters()).data
        self.hidden_state = (Variable(weight.new(self.config.hidden_size).zero_()),
                                Variable(weight.new(self.config.hidden_size).zero_()))

        if self.config.cuda:
            self.hidden_state[0] = self.hidden_state[0].cuda(self.config.gpuid)
            self.hidden_state[1] = self.hidden_state[1].cuda(self.config.gpuid)
