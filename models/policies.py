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
        embeds = self.encoder(state[-1])
        self.hidden_state = self.hidden_layer(embeds, self.hidden_state)
        out = self.decoder(self.hidden_state)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def init_hidden_state(self):
        weight = next(self.parameters()).data
	self.hidden_state = Variable(weight.new(self.config.hidden_size).zero_())

        if self.config.cuda:
            self.hidden_state = self.hidden_state.cuda(self.config.gpuid)


class LSTMPolicy(LanguageBasePolicy):
    def __init__(self, config, vocab_size):
   	LanguageBasePolicy.__init__(self, config, vocab_size)
        self.hidden_layer = nn.GRUCell(self.config.embedding_size, self.config.hidden_size)
	self.init_hidden_state()

    def forward(self, state):
        import pdb;pdb.set_trace()
        embeds = self.encoder(state[-1])
        self.hidden_state = self.hidden_layer(embeds, self.hidden_state)
        out = self.decoder(self.hidden_state)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    # def forward(self, state):
        # import pdb;pdb.set_trace()
        # embeds = self.encoder(state[-1])
        # self.hidden_state = self.hidden_layer(embeds, self.hidden_state)
        # out = self.decoder(self.hidden_state[0])
        # log_probs = F.log_softmax(out, dim=1)
        # return log_probs

    # def init_hidden_state(self):

        # weight = next(self.parameters()).data
        # h0, c0 = (Variable(weight.new(self.config.hidden_size).zero_()),
                                # Variable(weight.new(self.config.hidden_size).zero_()))

        # if self.config.cuda:
            # h0 = h0.cuda(self.config.gpuid)
            # c0 = c0.cuda(self.config.gpuid)

        # self.hidden_state = (h0, c0)

    def init_hidden_state(self):
        weight = next(self.parameters()).data
	self.hidden_state = Variable(weight.new(self.config.hidden_size).zero_())

        if self.config.cuda:
            self.hidden_state = self.hidden_state.cuda(self.config.gpuid)

class NgramPolicy(LanguageBasePolicy):
    def __init__(self, config, dictionary):
        LanguageBasePolicy.__init__(self, config, len(dictionary))
        self.lm = LM(self.config.lm_path)
        self.dictionary = dictionary

    def forward(self, state):
        log_probs = []
        context = [self.dictionary.idx2word[idx] for idx in reversed(state.data.tolist())]
        for w in self.dictionary.idx2word:
            log_probs.append(self.lm.logprob_strings(w, context))

        log_prob_tensor = torch.Tensor(log_probs)
        if self.config.cuda:
            log_prob_tensor = log_prob_tensor.cuda(self.config.gpuid)
        return Variable(log_prob_tensor, requires_grad=False)

    def init_hidden_state(self):
        pass

class DistributedBasePolicy(LanguageBasePolicy):
    def __init__(self, config, dictionary):
        LanguageBasePolicy.__init__(self, config, len(dictionary))
        self.lm = torch.load(self.config.lm_path)
        if self.config.cuda:
            self.lm = self.lm.cuda(self.config.gpuid)
        self.lm.eval()

    def forward(self, state):
        hidden = self.lm.init_hidden(state.shape[1])
        output, hidden = self.lm(state, hidden)
        return F.log_softmax(Variable(output[-1].data, requires_grad=False), dim=1)

    def init_hidden_state(self):
        pass
