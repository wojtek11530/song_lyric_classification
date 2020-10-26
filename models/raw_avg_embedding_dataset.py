from typing import Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from models.label_encoder import label_encoder


class RawAverageEmbeddingDataset(Dataset):
    def __init__(self, filepath: str):
        df = pd.read_csv(filepath, index_col=0)

        self.embeddings = np.array(df.loc[:, df.columns != 'emotion_4Q'])
        self.emotion_data = label_encoder.transform(df['emotion_4Q']).astype(np.int64)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        return self.embeddings[index, :], self.emotion_data[index]

    def __len__(self) -> int:
        return len(self.emotion_data)
