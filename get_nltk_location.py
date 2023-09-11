import nltk
import os
import shutil

# Path to the 'words' corpus
path_to_words = nltk.data.find('corpora/words')
print(path_to_words)

# Delete the directory
if os.path.exists(path_to_words):
    shutil.rmtree(path_to_words)