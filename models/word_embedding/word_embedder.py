import os

import numpy as np
from gensim.models.fasttext import load_facebook_model

from utils.singleton import Singleton

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'saved_models',
                           'fasttext_model_200_large_stopwords_removed.bin')


class WordEmbedder(metaclass=Singleton):
    def __init__(self):
        self._model = load_facebook_model(_MODEL_PATH)
        pass

    def __getitem__(self, word: str) -> np.ndarray:
        return self._model.wv[word]
