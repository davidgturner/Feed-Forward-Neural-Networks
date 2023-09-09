# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *
from torch.nn.utils.rnn import pad_sequence

class FFNN(nn.Module):

    def __init__(self, vocabulary_size, embedding_dimension, num_classes, word_embeddings):
        super(FFNN, self).__init__()

        # Embedding layer
        self.embedding = word_embeddings.get_initialized_embedding_layer()

        # Freeze embeddings
        for param in self.embedding.parameters():
            param.requires_grad = False

        self.g = nn.ReLU()  # activation function
        self.W = nn.Linear(embedding_dimension, num_classes)  # prediction layer after averaging embeddings
        self.log_softmax = nn.LogSoftmax(dim=1)  # output log probabilities over class labels
        nn.init.xavier_uniform_(self.W.weight)  # Xavier Glorot weight initialization

    def forward(self, x):
        # Handle unknown words
        x[x == -1] = 1  # Assuming 1 is the index of the UNK token

        # Add batch dimension
        x = x.unsqueeze(0)

        # Debug: print out-of-range indices
        out_of_range_indices = x[x >= self.embedding.weight.size(0)]
        if len(out_of_range_indices) > 0:
            print("Out of range indices detected:", out_of_range_indices)

        x = self.embedding(x)
        x = x.mean(dim=1)  # averaging embeddings
        x = self.g(x)
        return self.log_softmax(self.W(x))


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


import nltk
from nltk.corpus import words, reuters
from collections import Counter

class CustomSpellChecker:
    def __init__(self):
        nltk.download('reuters')
        self.word_freqs = Counter(reuters.words())
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.known_words = set(words.words())

    def __edits(self, word):
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in self.alphabet]
        inserts = [L + c + R for L, R in splits for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def __get_known_words(self, words):
        return set(word for word in words if word in self.known_words)

    def correct(self, word):
        if word in self.known_words:
            return word
        candidates = self.__get_known_words(self.__edits(word))
        if candidates:
            return max(candidates, key=self.word_freqs.get)
        return word


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    def __init__(self, ffnn, word_embeddings):
        self.ffnn = ffnn
        self.word_embeddings = word_embeddings
        self.spell_checker = CustomSpellChecker()
    
    def predict(self, ex_words, has_typos):
        if has_typos:
            ex_words = [self.spell_checker.correct(word) for word in ex_words]
        ex_tensor = torch.tensor([self.word_embeddings.word_indexer.index_of(word) for word in ex_words])
        
        device = next(self.ffnn.parameters()).device
        ex_tensor = ex_tensor.to(device)

        #ex_tensor = ex_tensor.to(device=self.ffnn.device) 
        logits = self.ffnn(ex_tensor.unsqueeze(0))
        return torch.argmax(logits).item()

def form_input(x) -> torch.Tensor:
    return x.long()

def train_deep_averaging_network(args, train_exs, dev_exs, word_embeddings: WordEmbeddings, train_model_for_typo_setting):
    train_xs = [torch.tensor([word_embeddings.word_indexer.index_of(word) for word in ex.words]) for ex in train_exs]
    train_ys = torch.tensor([ex.label for ex in train_exs])

    vocab_size = len(word_embeddings.vectors)
    embedding_dim = word_embeddings.get_embedding_length()

    num_classes = 2
    ffnn = FFNN(vocab_size, embedding_dim, num_classes, word_embeddings)
    optimizer = torch.optim.Adam(ffnn.parameters())

    criterion = nn.NLLLoss()

    for epoch in range(0, args.num_epochs):
        total_loss = 0.0
        for idx, x in enumerate(train_xs):
            x = form_input(x)
            # y = train_ys[idx]
            y = train_ys[idx].view(1)  # reshaping y to be 1D

            # print(x.shape)  # debugging the shape
            # print(y.shape)  # debugging the shape

            ffnn.zero_grad()

            #print(x.shape)  # Should ideally be something like [batch_size, sequence_length]
            #print(y.shape)  # Should ideally be [batch_size]

            probs = ffnn(x)
            loss = criterion(probs, y)
            loss.backward()

            total_loss += loss.item()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {total_loss}")
    train_correct = 0
    for idx in range(0, len(train_xs)):
        x = form_input(train_xs[idx])
        y = train_ys[idx]
        log_probs = ffnn.forward(x)
        prediction = torch.argmax(log_probs)
        if y == prediction:
            train_correct += 1
        print("Example " + repr(train_xs[idx]) + "; gold = " + repr(train_ys[idx]) + "; pred = " +\
              repr(prediction) + " with probs " + repr(probs))
    print(repr(train_correct) + "/" + repr(len(train_ys)) + " correct after training")

    return NeuralSentimentClassifier(ffnn, word_embeddings)