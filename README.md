# Feed-Forward-Neural-Networks
feed forward neural network approach for sentiment classification

# create virtual network 

``` python -m venv feed-fwd-nn_venv ```
``` .\feed-fwd-nn_venv\Scripts\activate ```

For PyTorch:
``` pip3 install torch torchvision torchaudio ```

# How to run

``` python optimization.py --lr 1  ```
``` python neural_sentiment_classifier.py --model TRIVIAL --no_run_on_test ```
``` python neural_sentiment_classifier.py --word_vecs_path data/glove.6B.300d-relativized.txt ```