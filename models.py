# models.py

import copy
from typing import Set
import torch
import torch.nn as nn
from sentiment_data import *
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

import nltk
from nltk.corpus import words, movie_reviews

UNK = 1
PAD = 0

class DeepAveragingNetwork(nn.Module):

    def __init__(self, embedding_dimension, num_classes, word_embeddings):
        super(DeepAveragingNetwork, self).__init__()

        # the first layer is the embedding layer
        self.embedding = word_embeddings.get_initialized_embedding_layer()

        # freeze the embeddings
        for param in self.embedding.parameters():
            param.requires_grad = False

        # hidden layer
        hidden_layer_size = 225
        self.hidden = nn.Linear(embedding_dimension, hidden_layer_size)
        nn.init.xavier_uniform_(self.hidden.weight)  # xavier glorot weight initialization for hidden layer

        # dropout rate after hidden layer
        drop_out_rate = 0.175
        self.dropout1 = nn.Dropout(drop_out_rate)

        # prediction layer after the hidden layer
        self.W = nn.Linear(hidden_layer_size, num_classes)
        nn.init.xavier_uniform_(self.W.weight)  # xavier glorot weight initialization for final linear layer

        # activation function
        self.g = nn.ReLU()

        # output log probabilities over class labels
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # handle any unknown words
        x[x == -1] = UNK  # 1 is the index of the UNK token

        x = self.embedding(x)
        x = x.mean(dim=1)  # averaging embeddings
        x = self.g(self.hidden(x))  # activation after hidden layer
        x = self.dropout1(x)  # dropout after hidden layer
        x = self.W(x)  # linear layer before softmax
        return self.log_softmax(x)

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

class SpellingCorrector:
    def __init__(self, words : Set[str] = set()):
        self.known_words = set(words)
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'

    def __edits(self, word: str):
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in self.alphabet]
        inserts = [L + c + R for L, R in splits for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def __get_known_words(self, words: Set):
        return set(word for word in words if word in self.known_words)

    def correct(self, word: str):
        if word in self.known_words:
            return word
        candidates = self.__get_known_words(self.__edits(word))
        if candidates:
            return next(iter(candidates))
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

        # use english words and movie reviews to feed spelling corrector
        nltk.download("words")
        nltk.download('movie_reviews')

        all_known_words = set()
        eng_words_set = set(words.words())
        movie_reviews_set = set(movie_reviews.words())
        all_known_words.update(eng_words_set)
        all_known_words.update(movie_reviews_set)

        self.spelling_corrector = SpellingCorrector(all_known_words)

    def train(self, X_train, y_train, X_dev, y_dev, args):
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(self.ffnn.parameters(), lr=args.lr)
        best_dev_accuracy = 0.0
        best_model = None

        # logic for patience which is how many epochs to wait after it does not improve
        wait = 0
        epochs_without_improvement = 2 

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)

        for epoch in range(0, args.num_epochs):
            total_loss = 0.0
            for x_batch, y_batch in train_loader:
                self.ffnn.zero_grad()
                probs = self.ffnn(x_batch)
                loss = criterion(probs, y_batch)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()

            # evaluate the dev set after each epoch
            dev_accuracy = self.eval_prediction(X_dev, y_dev)
            print(f"Epoch {epoch}, Loss: {total_loss}, Dev Accuracy: {dev_accuracy}")
            
            # check if the dev accuracy improved
            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                best_model = copy.deepcopy(self.ffnn)
                wait = 0  # reset wait
            else:
                wait += 1

            # break early if no improvement
            if wait >= epochs_without_improvement:
                print(f"Breaking due to no improvement after {wait} epochs.")
                break

        # revert to the best model once training done
        self.ffnn = best_model

    # eval_prediction - computes the predictions for X and checks how many match y
    def eval_prediction(self, X, y):
        with torch.no_grad():
            logits = self.ffnn(X)
            predicted = torch.argmax(logits, dim=1)
            correct = (predicted == y).float().sum().item()
            accuracy = correct / len(y)
        return accuracy

    def predict(self, ex_words, has_typos):
        if has_typos:
            ex_words = [self.spelling_corrector.correct(word) for word in ex_words]

        ex_tensor = torch.tensor([self.word_embeddings.word_indexer.index_of(word) for word in ex_words])
        
        device = next(self.ffnn.parameters()).device
        ex_tensor = ex_tensor.to(device)

        logits = self.ffnn.forward(ex_tensor.unsqueeze(0))

        prediction = torch.argmax(logits).item()

        return prediction

def form_input(x) -> torch.Tensor:
    return x.long()

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    return xx_pad, torch.tensor(yy)

def pad_tensor(tensor_list, pad=0):
    max_len = max(tensor.size(0) for tensor in tensor_list)
    out_dims = (len(tensor_list), max_len)
    out_tensor = tensor_list[0].data.new(*out_dims).fill_(pad)
    for i, tensor in enumerate(tensor_list):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
    return out_tensor

def train_deep_averaging_network(args, train_exs, dev_exs, word_embeddings: WordEmbeddings, train_model_for_typo_setting):

    # convert to tensors
    train_xs = [torch.tensor([word_embeddings.word_indexer.index_of(word) for word in ex.words]) for ex in train_exs]
    train_ys = torch.tensor([ex.label for ex in train_exs])
    padded_train_xs = pad_tensor(train_xs)

    dev_xs = [torch.tensor([word_embeddings.word_indexer.index_of(word) for word in ex.words]) for ex in dev_exs]
    dev_ys = torch.tensor([ex.label for ex in dev_exs])
    padded_dev_xs = pad_tensor(dev_xs)

    # initialize the network
    num_classes = 2
    embedding_dim = word_embeddings.get_embedding_length()
    ffnn = DeepAveragingNetwork(embedding_dim, num_classes, word_embeddings)

    # create the classifier and train it
    classifier = NeuralSentimentClassifier(ffnn, word_embeddings)
    classifier.train(padded_train_xs, train_ys, padded_dev_xs, dev_ys, args)

    return classifier