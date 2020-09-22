from typing import Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from models.label_encoder import label_encoder
from models.word_embedding.word_embedder import WordEmbedder


class LyricsDataset(Dataset):
    def __init__(self, filepath: str):
        dataset = pd.read_csv(filepath, index_col=0)

        emotion = dataset['emotion_4Q']
        self.emotion_data = label_encoder.transform(emotion)
        self.lyrics_data = dataset['lyrics'].values

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
