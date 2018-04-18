import os
import torch

class Dictionary(object):
    def __init__(self, bos='<s>', eos='</s>'):
        self.word2idx = {bos : 0, eos : 1}
        self.idx2word = [bos, eos]

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, bos='<s>', eos='</s>'):
        self.dictionary = Dictionary(bos, eos)
        self.train = self.tokenize(os.path.join(path, 'train.txt'), bos, eos)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), bos, eos)
        self.test = self.tokenize(os.path.join(path, 'test.txt'), bos, eos)

    def tokenize(self, path, bos, eos):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = [bos] + line.split() + [eos]
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = [bos] + line.split() + [eos]
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
