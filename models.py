# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *

class FFNN(nn.Module):

    def __init__(self, vocabulary_size, embedding_dimension, num_classes):
        super(FFNN, self).__init__()

        #  embedding layer
        # TODO - do some averaging as this is a deep averaging network (bag of vectors)
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        
        # linear layer for prediction
        # self.fc = nn.Linear(embedding_dimension, num_classes)

        # input is from last layer
        self.V = nn.Linear(vocabulary_size, embedding_dimension)
        self.g = nn.ReLU() # nn.Tanh()
        self.W = nn.Linear(embedding_dimension, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=0)
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)
        return self.log_softmax(self.fc(x), dim=1)
        # return self.log_softmax(self.W(self.g(self.V(x))))
        # return self.log_softmax(self.W(self.g(self.V(x))))


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
    """
    Form the input to the neural network. In general this may be a complex function that synthesizes multiple pieces
    of data, does some computation, handles batching, etc.

    :param x: a [num_samples x inp] numpy array containing input data
    :return: a [num_samples x inp] Tensor
    """
    return torch.from_numpy(x).float()

def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """
    # Synthetic data for XOR: y = x0 XOR x1
    #train_xs = np.array([[0, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 1]], dtype=np.float32)
    #train_ys = np.array([0, 1, 1, 1, 1, 0], dtype=np.float32)

    train_xs = [[word_embeddings.get_embedding(word) for word in ex.words] for ex in train_exs]
    train_xs = np.array([train_xs])

    train_ys = [ex.label for ex in train_exs]
    train_ys = np.array([train_ys])

    # dev_indices = [[word_embeddings.get_embedding(word) for word in ex.words] for ex in dev_exs]
    # dev_labels = [ex.label for ex in dev_exs]

    # Define some constants
    # Inputs are of size 2
    feat_vec_size = 2
    # Let's use 10 hidden units
    embedding_size = 10
    # We're using 2 classes. What's presented here is multi-class code that can scale to more classes, though
    # slightly more compact code for the binary case is possible.
    num_classes = 2

    # RUN TRAINING AND TEST
    num_epochs = 100
    ffnn = FFNN(feat_vec_size, embedding_size, num_classes)
    initial_learning_rate = 0.1
    optimizer = optim.Adam(ffnn.parameters(), lr=initial_learning_rate)
    for epoch in range(0, num_epochs):
        ex_indices = [i for i in range(0, len(train_xs))]
        random.shuffle(ex_indices)
        total_loss = 0.0
        for idx in ex_indices:
            x = form_input(train_xs[idx])
            y = train_ys[idx]

            # Build one-hot representation of y. Instead of the label 0 or 1, y_onehot is either [0, 1] or [1, 0]. This
            # way we can take the dot product directly with a probability vector to get class probabilities.
            y_onehot = torch.zeros(num_classes)
            # scatter will write the value of 1 into the position of y_onehot given by y
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y,dtype=np.int64)), 1)
            # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
            ffnn.zero_grad()
            probs = ffnn.forward(x)
            # Can also use built-in NLLLoss as a shortcut here but we're being explicit here
            loss = torch.neg(probs).dot(y_onehot)
            total_loss += loss
            # Computes the gradient and takes the optimizer step
            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
    # Evaluate on the train set
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
    
    # # convert words to indices.
    # train_indices = [[word_embeddings.get_embedding(word) for word in ex.words] for ex in train_exs]
    # train_labels = [ex.label for ex in train_exs]

    # dev_indices = [[word_embeddings.get_embedding(word) for word in ex.words] for ex in dev_exs]
    # dev_labels = [ex.label for ex in dev_exs]

    # vocab_size = len(word_embeddings.vectors)
    # embedding_dim = word_embeddings.get_embedding_length()

    # # instantiate model
    # model = FeedForwardTextClassificationModel(vocabulary_size=vocab_size, embedding_dimension=embedding_dim, num_classes=2)
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # # train the model
    # for epoch in range(args.num_epochs):
    #     total_loss = 0
    #     model.train()
    #     for words, label in zip(train_indices, train_labels):
    #         optimizer.zero_grad()
            
    #         inputs = torch.tensor(words).long()
    #         label = torch.tensor([label]).long()
            
    #         # log_probs = model(inputs)
    #         model.zero_grad()
    #         log_probs = model.forward(inputs)
            
    #         loss = criterion(log_probs, label)
    #         loss.backward()
    #         optimizer.step()
            
    #         total_loss += loss.item()

    #     # evaluate on dev set
    #     model.eval()
    #     correct = 0
    #     with torch.no_grad():
    #         for words, label in zip(dev_indices, dev_labels):
    #             inputs = torch.tensor(words).long()
    #             label = torch.tensor([label]).long()
                
    #             log_probs = model(inputs)
    #             _, predicted = torch.max(log_probs, 1)
                
    #             if predicted.item() == label.item():
    #                 correct += 1

    #     accuracy = correct / len(dev_exs)
    #     print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {total_loss}, Dev Accuracy: {accuracy}")

    # return NeuralSentimentClassifier(model)

