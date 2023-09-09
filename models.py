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

        # embedding layer will convert word indices into their respective embeddings.
        # embeddings for each sequence will be averaged.
        # averaged embedding will be passed through your feed-forward layers.
        self.embedding = word_embeddings.get_initialized_embedding_layer()

        # freeze embeddings
        for param in self.embedding.parameters():
            param.requires_grad = False

        self.g = nn.ReLU() # activation function
        self.W = nn.Linear(embedding_dimension, num_classes)  # linear layer for prediction after averaging embeddings
        self.log_softmax = nn.LogSoftmax(dim=1)  # output log probabilities over class labels
        nn.init.xavier_uniform_(self.W.weight)  # Initialize weights according to a formula due to Xavier Glorot.

    def forward(self, x):
        print("FFNN forward - type ", x.dtype)
        print("x shape before embedding layer call ", x.shape)
        x = self.embedding(x)
        print("x shape after embedding layer call ", x.shape)
        x = x.mean(dim=1)  # averaging embeddings
        x = x.view(x.size(0), -1)
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


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    def __init__(self, ffnn , word_embedding):
        self.ffnn = ffnn
        self.word_embedding = word_embedding
    
    def predict(self, x):
        # TODO - should look like old proj
        raise NotImplementedError

def form_input(x) -> torch.Tensor:
    return x.long()

def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    
    vocabulary_size = len(word_embeddings.vectors)
    train_xs = []
    # words with an index of -1 are out-of-vocabulary (OOV) words.
    # UNK (Unknown) token is to handle such OOV words.
    UNK_INDEX = 1

    train_xs = []
    for ex in train_exs:
        indices = []
        for word in ex.words:
            idx = word_embeddings.word_indexer.index_of(word)
            if idx == -1: # OOV word
                idx = UNK_INDEX
            indices.append(idx)
        train_xs.append(torch.tensor(indices))

    # train_xs = [torch.tensor([word_embeddings.word_indexer.index_of(word) for word in ex.words]) for ex in train_exs]
    train_xs = [torch.tensor([word_embeddings.word_indexer.index_of(word) for word in ex.words]) for ex in train_exs]
    train_xs_padded = pad_sequence(train_xs, batch_first=True)

    train_ys = torch.tensor([ex.label for ex in train_exs])

    embedding_size = 10
    num_classes = 2
    num_epochs = 100
    
    ffnn = FFNN(vocabulary_size, embedding_size, num_classes, word_embeddings)

    initial_learning_rate = 0.1

    # define Adam optimizer
    optimizer = optim.Adam(ffnn.parameters(), lr=initial_learning_rate)

    # NLLLoss expects log probabilities and model provides them with nn.LogSoftmax
    criterion = nn.NLLLoss()
    
    for epoch in range(0, num_epochs):
        ex_indices = list(range(len(train_xs_padded)))
        random.shuffle(ex_indices)
        total_loss = 0.0
        for idx in ex_indices:
            x = form_input(train_xs_padded[idx])
            y = train_ys[idx]

            ffnn.zero_grad()

            print(torch.max(x))  # check if there's any value in x that is equal to or greater than your vocabulary_size. This will directly lead to the index out of range error.
            probs = ffnn.forward(x)

            loss = criterion(probs, y)

            total_loss += loss
            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
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

    return NeuralSentimentClassifier(model, word_embeddings)