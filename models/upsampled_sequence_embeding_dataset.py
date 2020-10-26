from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from nltk import word_tokenize
from torch.utils.data import Dataset

from models.label_encoder import label_encoder
from models.word_embedding.word_embedder import WordEmbedder
from preprocessing.text_preprocessor import preprocess, remove_stop_words, lemmatize_text

_RANDOM_SEED = 42


class UpsampledSequenceEmbeddingDataset(Dataset):
    def __init__(self, filepath: str, embedding_dim: int, max_num_words: int, removing_stop_words: bool = False,
                 lemmatization: bool = False):
        self._embedding_dim = embedding_dim
        self._max_words_num = max_num_words

        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Upsampled average embedding dataset creation started")
        song_df = pd.read_csv(filepath, index_col=0)
        song_df = self._preprocess_lyrics_in_df(song_df, lemmatization, removing_stop_words)

        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Getting sequence of embedding started")
        X = self._get_sequence_of_embeddings_array(song_df)
        y = np.array(song_df.loc[:, song_df.columns == 'emotion_4Q'])
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Getting sequence of embedding ended")

        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Upsampling with SMOTE started")
        X_upsampled, y_upsampled = self._get_upsampled_data_with_smote(X, y)
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Upsampling with SMOTE ended")

        self.embeddings = X_upsampled
        self.emotion_data = label_encoder.transform(y_upsampled).astype(np.int64)
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Upsampled average embedding dataset creation ended")

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        return self.embeddings[index, :].reshape((-1, self._embedding_dim)), self.emotion_data[index]

    def __len__(self) -> int:
        return len(self.emotion_data)

    def _preprocess_lyrics_in_df(self, song_df: pd.DataFrame, lemmatization: bool, removing_stop_words: bool):
        song_df['lyrics'] = song_df['lyrics'].apply(
            lambda x: preprocess(x, remove_punctuation=True, remove_text_in_brackets=True))

        if removing_stop_words:
            song_df['lyrics'] = song_df['lyrics'].apply(lambda x: remove_stop_words(x))

        if lemmatization:
            song_df['lyrics'] = song_df['lyrics'].apply(lambda x: lemmatize_text(x))

        song_df = song_df[song_df['lyrics'] != '']
        return song_df

    def _get_upsampled_data_with_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        smote = SMOTE(random_state=_RANDOM_SEED)
        X_upsampled, y_upsampled = smote.fit_sample(X, y.ravel())
        return X_upsampled, y_upsampled

    def _get_sequence_of_embeddings_array(self, df: pd.DataFrame) -> np.ndarray:
        word_embedder = WordEmbedder()

        X = np.zeros(shape=(len(df), self._max_words_num * self._embedding_dim), dtype=np.float32)

        for line_number, (index, row) in enumerate(df.iterrows()):
            words = word_tokenize(row['lyrics'])
            words = words[:self._max_words_num]
            embeddings = np.array([word_embedder[word] for word in words])
            flatten_embeddings = embeddings.flatten()

            limit = min(len(flatten_embeddings), self._max_words_num * self._embedding_dim)
            X[line_number, 0:limit] = flatten_embeddings[:limit]

        return X
