from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from nltk import word_tokenize
from torch.utils.data import Dataset

from models.label_encoder import label_encoder
from models.word_embedding.word_embedder import WordEmbedder
from preprocessing.text_preprocessor import lemmatize_text, preprocess, remove_stop_words

_RANDOM_SEED = 42


class UpsampledAverageEmbeddingDataset(Dataset):
    def __init__(self, filepath: str, embedding_dim: int, removing_stop_words: bool = False,
                 lemmatization: bool = False):
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Upsampled average embedding dataset creation started")
        song_df = pd.read_csv(filepath, index_col=0)
        song_df = self._preprocess_lyrics_in_df(song_df, lemmatization, removing_stop_words)
        song_df = self._get_df_with_embeddings(embedding_dim, song_df)
        X_upsampled, y_upsampled = self.get_upsampled_data_with_smote(song_df)
        self.embeddings = X_upsampled
        self.emotion_data = label_encoder.transform(y_upsampled).astype(np.int64)
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Upsampled average embedding dataset creation ended")

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        return self.embeddings[index, :], self.emotion_data[index]

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

    def get_upsampled_data_with_smote(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = np.array(df.loc[:, df.columns != 'emotion_4Q'])
        y = np.array(df.loc[:, df.columns == 'emotion_4Q'])
        smote = SMOTE(random_state=_RANDOM_SEED)
        X_upsampled, y_upsampled = smote.fit_sample(X, y.ravel())
        return X_upsampled, y_upsampled

    def _get_df_with_embeddings(self, embedding_dim, song_df):
        word_embedder = WordEmbedder()
        for i in range(embedding_dim):
            song_df['emb_' + str(i)] = None

        for line_number, (index, row) in enumerate(song_df.iterrows()):
            text = row['lyrics']
            words = word_tokenize(text)
            embeddings = np.array([word_embedder[word] for word in words])
            avg_embedding = np.mean(embeddings, axis=0)
            for i in range(embedding_dim):
                song_df.at[index, 'emb_' + str(i)] = avg_embedding[i]

        song_df = song_df.drop(
            ['arousal_mean', 'arousal_std', 'artist', 'dataset', 'emotion_2Q', 'general_genre', 'genre',
             'language', 'lyrics', 'song_id_from_src', 'title', 'valence_mean', 'valence_std'], axis=1)
        return song_df
