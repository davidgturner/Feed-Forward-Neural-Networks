# models.py

import copy
import torch
import torch.nn as nn
from sentiment_data import *
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

import nltk
from nltk.corpus import words, movie_reviews
from nltk.metrics.distance import edit_distance

nltk.download('words', quiet=True)
nltk.download('movie_reviews', quiet=True)

UNK = 1
PAD = 0

class FFNN(nn.Module):

    def __init__(self, embedding_dimension, num_classes, word_embeddings):
        super(FFNN, self).__init__()

        # the first layer is the embedding layer
        self.embedding = word_embeddings.get_initialized_embedding_layer()

        # freeze the embeddings
        for param in self.embedding.parameters():
            param.requires_grad = False

        # dropout rate after averaging embeddings
        drop_out_rate = 0.175

        # hidden layer
        hidden_layer_size = 225
        self.hidden = nn.Linear(embedding_dimension, hidden_layer_size)
        nn.init.xavier_uniform_(self.hidden.weight)  # xavier glorot weight initialization for hidden layer

        # dropout rate after hidden layer
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

class SpellChecker:
    def __init__(self, words = set(movie_reviews.words())):
        self.known_words = words
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'

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
        self.wordlist = set(movie_reviews.words())
        self.spell_checker = SpellChecker(self.wordlist)

    def train(self, X_train, y_train, X_dev, y_dev, args):
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(self.ffnn.parameters(), lr=args.lr)
        best_dev_accuracy = 0.0
        best_model = None

        # logic for patience
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

            # Evaluate on the dev set at the end of each epoch
            dev_accuracy = self.evaluate(X_dev, y_dev)
            print(f"Epoch {epoch}, Loss: {total_loss}, Dev Accuracy: {dev_accuracy}")
            
            # Check if the dev accuracy improved
            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                best_model = copy.deepcopy(self.ffnn)
                wait = 0  # reset wait
            else:
                wait += 1

            # Break early if we waited for a certain number of epochs without improvement
            if wait >= epochs_without_improvement:
                print(f"Breaking due to no improvement after {wait} epochs.")
                break

        # Once training is done, revert to the best model
        self.ffnn = best_model

    # evaluate - computes the predictions for X and checks how many match y
    def evaluate(self, X, y):
        with torch.no_grad():
            logits = self.ffnn(X)
            predicted = torch.argmax(logits, dim=1)
            correct = (predicted == y).float().sum().item()
            accuracy = correct / len(y)
        return accuracy

    def predict(self, ex_words, has_typos):
        if has_typos:
            ex_words = [self.spell_checker.correct(word) for word in ex_words]
    
        ex_tensor = torch.tensor([self.word_embeddings.word_indexer.index_of(word) for word in ex_words])
        
        device = next(self.ffnn.parameters()).device
        ex_tensor = ex_tensor.to(device)

        logits = self.ffnn.forward(ex_tensor.unsqueeze(0))

        return torch.argmax(logits).item()

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

spell_checker = SpellChecker()

def train_deep_averaging_network(args, train_exs, dev_exs, word_embeddings: WordEmbeddings, train_model_for_typo_setting):
    # If train_model_for_typo_setting is True, we spell-correct the words in train_exs
    if train_model_for_typo_setting:
        for ex in train_exs:
            ex.words = [spell_checker.correct(word) for word in ex.words]

    # Convert examples to tensor representations
    train_xs = [torch.tensor([word_embeddings.word_indexer.index_of(word) for word in ex.words]) for ex in train_exs]
    train_ys = torch.tensor([ex.label for ex in train_exs])
    padded_train_xs = pad_tensor(train_xs)

    dev_xs = [torch.tensor([word_embeddings.word_indexer.index_of(word) for word in ex.words]) for ex in dev_exs]
    dev_ys = torch.tensor([ex.label for ex in dev_exs])
    padded_dev_xs = pad_tensor(dev_xs)

    # Initialize the FFNN model
    embedding_dim = word_embeddings.get_embedding_length()
    num_classes = 2
    ffnn = FFNN(embedding_dim, num_classes, word_embeddings)

    # Create the NeuralSentimentClassifier and train it
    classifier = NeuralSentimentClassifier(ffnn, word_embeddings)
    classifier.train(padded_train_xs, train_ys, padded_dev_xs, dev_ys, args)

    return classifier


# def train_deep_averaging_network(args, train_exs, dev_exs, word_embeddings: WordEmbeddings, train_model_for_typo_setting):
    
#     # If train_model_for_typo_setting is True, we spell-correct the words in train_exs
#     if train_model_for_typo_setting:
#         for ex in train_exs:
#             ex.words = [spell_checker.correct(word) for word in ex.words]

#     train_xs = [torch.tensor([word_embeddings.word_indexer.index_of(word) for word in ex.words]) for ex in train_exs]
#     train_ys = torch.tensor([ex.label for ex in train_exs])
    
#     padded_train_xs = pad_tensor(train_xs)

#     embedding_dim = word_embeddings.get_embedding_length()

#     num_classes = 2
#     ffnn = FFNN(embedding_dim, num_classes, word_embeddings)
#     optimizer = torch.optim.Adam(ffnn.parameters(), lr=args.lr)
#     criterion = nn.NLLLoss()

#     best_dev_accuracy = 0
#     epoch_improvement_waiting_period = 3
#     wait = 0

#     train_dataset = TensorDataset(padded_train_xs, train_ys)
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)

#     for epoch in range(0, args.num_epochs):
#         total_loss = 0.0
#         for x, y in train_loader:
#             ffnn.zero_grad()
#             probs = ffnn(x)
#             loss = criterion(probs, y)
#             loss.backward()
#             total_loss += loss.item()
#             optimizer.step()
#         print(f"Epoch {epoch}, Loss: {total_loss}")

#         # evaluate the dev set
#         dev_correct = 0
#         for ex in dev_exs:
#             words = [word for word in ex.words]
#             x = torch.tensor([word_embeddings.word_indexer.index_of(word) for word in words]).unsqueeze(0)
#             y = ex.label
#             log_probs = ffnn(x)
#             prediction = torch.argmax(log_probs)
#             if y == prediction:
#                 dev_correct += 1
#         dev_accuracy = dev_correct / len(dev_exs)
#         print(f"Epoch {epoch}, Dev Accuracy: {dev_accuracy:.4f}")

#         # check for early stopping
#         if dev_accuracy > best_dev_accuracy:
#             best_dev_accuracy = dev_accuracy
#             best_model = copy.deepcopy(ffnn)
#             wait = 0
#         else:
#             wait += 1
#             if wait > epoch_improvement_waiting_period:
#                 print("Early stopping triggered.")
#                 break

#     # use the best model
#     ffnn = best_model
#     train_correct = 0
#     for idx in range(0, len(train_xs)):
#         x = form_input(train_xs[idx])
#         y = train_ys[idx]

#         log_probs = ffnn.forward(x)

#         prediction = torch.argmax(log_probs)
#         if y == prediction:
#             train_correct += 1
    
#     print(repr(train_correct) + "/" + repr(len(train_ys)) + " correct after training")

#     return NeuralSentimentClassifier(ffnn, word_embeddings)