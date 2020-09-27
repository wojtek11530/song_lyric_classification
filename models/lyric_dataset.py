from typing import Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from models.label_encoder import label_encoder
from models.word_embedding.word_embedder import WordEmbedder
from preprocessing.text_preprocessor import preprocess, remove_stop_words


class LyricsDataset(Dataset):
    def __init__(self, filepath: str):
        song_df = pd.read_csv(filepath, index_col=0)

        song_df['lyrics'] = song_df.apply(
            lambda x: preprocess(x['lyrics'], remove_punctuation=True, remove_text_in_brackets=True),
            axis=1)
        # song_df['lyrics'] = song_df.apply(lambda x: remove_stop_words(x['lyrics']), axis=1)
        song_df = song_df[song_df['lyrics'] != '']

        self.emotion_data = label_encoder.transform(song_df['emotion_4Q'])
        self.lyrics_data = song_df['lyrics'].values
        self.word_embedder = WordEmbedder()

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        return self._get_embeddings(self.lyrics_data[index]), \
               self.emotion_data[index]

    def __len__(self) -> int:
        return len(self.emotion_data)

    def _get_embeddings(self, sentence: str) -> np.ndarray:
        words = sentence.split()
        embedding = np.array([self.word_embedder[word] for word in words])
        return embedding
