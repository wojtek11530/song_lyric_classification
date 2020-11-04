from typing import Tuple

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset

from models.label_encoder import label_encoder
from models.word_embedding.word_embedder import WordEmbedder
from preprocessing.text_preprocessor import lemmatize_text, preprocess, remove_stop_words


class LyricsDataset(Dataset):
    def __init__(self, song_df: pd.DataFrame, removing_stop_words: bool = False, lemmatization: bool = False):
        self._removing_stop_words = removing_stop_words
        self._lemmatization = lemmatization

        song_df = self._preprocess_lyrics_in_df(song_df)

        self.emotion_data = label_encoder.transform(song_df['emotion_4Q'])
        self.lyrics_data = song_df['lyrics'].values
        self.word_embedder = WordEmbedder()

    @classmethod
    def from_file(cls, filepath: str, removing_stop_words: bool = False, lemmatization: bool = False):
        song_df = pd.read_csv(filepath, index_col=0)
        class_instance = cls(song_df, removing_stop_words, lemmatization)
        return class_instance

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        return self._get_embeddings(self.lyrics_data[index]), \
               self.emotion_data[index]

    def __len__(self) -> int:
        return len(self.emotion_data)

    def _preprocess_lyrics_in_df(self, song_df: pd.DataFrame) -> pd.DataFrame:
        song_df['lyrics'] = song_df['lyrics'].apply(
            lambda x: preprocess(x, remove_punctuation=True, remove_text_in_brackets=True))

        if self._removing_stop_words:
            song_df['lyrics'] = song_df['lyrics'].apply(lambda x: remove_stop_words(x))

        if self._lemmatization:
            song_df['lyrics'] = song_df['lyrics'].apply(lambda x: lemmatize_text(x))

        song_df = song_df[song_df['lyrics'] != '']
        return song_df

    def _get_embeddings(self, text: str) -> np.ndarray:
        words = word_tokenize(text)
        embedding = np.array([self.word_embedder[word] for word in words])
        return embedding
