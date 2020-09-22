import os

import fasttext
import numpy as np

from utils.singleton import Singleton

_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'cc.en.300.bin')


class WordEmbedder(metaclass=Singleton):
    def __init__(self):
        self._model = fasttext.load_model(_MODEL_PATH)

    def __getitem__(self, word: str) -> np.ndarray:
        return self._model.get_word_vector(word)
