from typing import List, Set
import nltk
from nltk.corpus import words, movie_reviews
import difflib

from sentiment_data import read_sentiment_examples

# from nltk.data import find
# print("HEREHEHREHE ", find('corpora/words'))

from nltk.corpus import stopwords
import string

# Ensure the stopwords are downloaded
nltk.download('stopwords', quiet=True)
# stop_words = set(stopwords.words('english'))

import difflib


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
    

# class SpellChecker:
#     def __init__(self, words : List[str] = None):
#         if words is None:
#             self.known_words : Set = set(nltk.corpus.words.words())
#         else:
#             self.known_words : Set = set(words)
        
#         self.stop_words = set(stopwords.words('english'))

#     def correct(self, word_input: str):
#         # print("word_input ", word_input)
#         if word_input in self.known_words or len(word_input) <= 3 or word_input in self.stop_words or word_input in string.punctuation:
#             return word_input

#         # We initialize the SequenceMatcher with the word to be corrected.
#         matcher = difflib.SequenceMatcher(a=word_input)

#         # We then find the best match from the known words.
#         best_match = ""
#         threshold_ratio = 0.80
#         for known_word in self.known_words:
#             matcher.set_seq2(known_word)
#             ratio = matcher.ratio()
#             if ratio > threshold_ratio:
#                 # highest_ratio = ratio
#                 best_match = known_word
#                 return best_match

#         return word_input

        #print("fixed word ", word, " to ", best_match)

        # If we found a reasonably close match, return it. Otherwise, return the original word.
        #if highest_ratio > 0.8:  # This threshold can be adjusted based on your needs.
        #    return best_match
        #else:
        #    return word



# class SpellChecker:
#     def __init__(self, known_words=None):
#         nltk.download("words")
#         nltk.download('movie_reviews')

#         if known_words is None:
#             eng_words = set(words.words())
#             mr_words = set(movie_reviews.words())
#             self.known_words = eng_words
#             self.known_words.update(mr_words)
#         else:
#             self.known_words = known_words

#         self.cache = {}

#     def correct(self, word):
#         if word in self.known_words:
#             return word
        
#         if word in self.cache:
#             return self.cache[word]
        
#         close_words = difflib.get_close_matches(word, self.known_words, n=1)
        
#         if close_words:
#             self.cache[word] = close_words[0]
#         else:
#             self.cache[word] = word
        
#         print("fixed word ", word, " to ", self.cache[word])

#         return self.cache[word]



sc = SpellChecker()
ex_words = read_sentiment_examples("data/dev-typo.txt")
# sentences = [ex.words for ex in ex_words]
# fixed_words = [sc.correct(word) for word in sentences]
ex_words_str = [ex.words for ex in ex_words]
# print(ex_words_str) 


for sent in ex_words_str:
    print("old sent ", sent)
    for word in sent:
        word = sc.correct(word)
    print("new sent ", sent)
# fixed_sentences = [[sc.correct(word) for word in sentence] for sentence in sentences]
print(ex_words_str)

# word_to_fix = "couraae"
# cw = sc.correct(word_to_fix)
# print(word_to_fix)
# print(cw)
