import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from nltk import word_tokenize
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from models.base import BaseModel
from models.lyric_dataset import LyricsDataset
from models.word_embedding.word_embedder import WordEmbedder
from preprocessing.text_preprocessor import lemmatize_text, preprocess, remove_stop_words

_WORKERS_NUM = 4

_PROJECT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TRAIN_DATASET_FILEPATH = os.path.join(_PROJECT_DIRECTORY, 'datasets', 'train_dataset.csv')
_VAL_DATASET_FILEPATH = os.path.join(_PROJECT_DIRECTORY, 'datasets', 'val_dataset.csv')
_TEST_DATASET_FILEPATH = os.path.join(_PROJECT_DIRECTORY, 'datasets', 'test_dataset.csv')


def avg_embedding_collate(batch: List[Tuple[np.ndarray, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    xx, yy = zip(*batch)
    xx = torch.Tensor([np.mean(embeddings, axis=0) for embeddings in xx])
    yy = torch.Tensor(yy).to(dtype=torch.int64)
    return xx, yy


class MLPClassifier(BaseModel):
    def __init__(self, input_size: int = 100, output_size: int = 4,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-5, dropout: float = 0.5, batch_size: int = 128,
                 removing_stop_words: bool = False, lemmatization: bool = False):
        super(MLPClassifier, self).__init__()

        self._train_set: Optional[Dataset] = None
        self._val_set: Optional[Dataset] = None
        self._test_set: Optional[Dataset] = None

        layer_1_out_dim = 24
        self._layer_1 = torch.nn.Linear(input_size, layer_1_out_dim)
        self._layer_2 = torch.nn.Linear(layer_1_out_dim, output_size)
        self._dropout = torch.nn.Dropout(p=dropout)

        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._word_embedder = WordEmbedder()

        self._weight_decay = weight_decay
        self._removing_stop_words = removing_stop_words
        self._lemmatization = lemmatization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.view(x.size(0), -1)
        x = x.float()
        # layer 1
        x = self._layer_1(x)
        x = F.relu(x)
        x = self._dropout(x)
        # layer 2
        x = self._layer_2(x)
        return x

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), self._learning_rate, weight_decay=self._weight_decay)

    def train_dataloader(self) -> DataLoader:
        if self._train_set is None:
            self._train_set = LyricsDataset(_TRAIN_DATASET_FILEPATH,
                                            removing_stop_words=self._removing_stop_words,
                                            lemmatization=self._lemmatization)
        return DataLoader(self._train_set, batch_size=self._batch_size, shuffle=True,
                          drop_last=False, collate_fn=avg_embedding_collate)

    def val_dataloader(self) -> DataLoader:
        if self._val_set is None:
            self._val_set = LyricsDataset(_VAL_DATASET_FILEPATH,
                                          removing_stop_words=self._removing_stop_words,
                                          lemmatization=self._lemmatization)
        return DataLoader(self._val_set, batch_size=self._batch_size, drop_last=False,
                          collate_fn=avg_embedding_collate)

    def test_dataloader(self) -> DataLoader:
        if self._test_set is None:
            self._test_set = LyricsDataset(_TEST_DATASET_FILEPATH,
                                           removing_stop_words=self._removing_stop_words,
                                           lemmatization=self._lemmatization)
        return DataLoader(self._test_set, batch_size=self._batch_size, drop_last=False,
                          collate_fn=avg_embedding_collate)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) \
            -> Dict[str, Any]:
        x, y_labels = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y_labels)
        total = len(y_labels)
        correct = self._get_correct_prediction_count(logits, y_labels)
        return {'loss': loss, "correct": correct, "total": total, 'log': {'train_loss': loss}}

    def training_epoch_end(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])

        tensorboard_logs = {'loss': avg_loss, "train_acc": correct / total}
        return {'loss': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, val_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) \
            -> Dict[str, Any]:
        x, y_labels = val_batch
        logits = self(x)
        loss = F.cross_entropy(logits, y_labels)
        total = len(y_labels)
        correct = self._get_correct_prediction_count(logits, y_labels)
        return {'val_loss': loss, "correct": correct, "total": total}

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) \
            -> Dict[str, Any]:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])

        tensorboard_logs = {'val_loss': avg_loss, "val_acc": correct / total}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def _get_correct_prediction_count(self, logits: torch.Tensor, y_labels: torch.Tensor) -> int:
        probs = torch.softmax(logits, dim=1)
        return int(probs.argmax(dim=1).eq(y_labels).sum().item())

    def _batch_step(self, batch: List) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y_labels = batch
        logits = self(x)
        _, y_hat = torch.max(logits, dim=1)
        return y_labels, y_hat

    def predict(self, lyrics: str) -> np.ndarray:
        lyrics = preprocess(lyrics, remove_punctuation=True, remove_text_in_brackets=True)
        if self._removing_stop_words:
            lyrics = remove_stop_words(lyrics)
        if self._lemmatization:
            lyrics = lemmatize_text(lyrics)

        avg_embedding = self._get_avg_embedding(lyrics)
        res = self(avg_embedding)
        probs = torch.softmax(res, dim=-1)
        label = probs.argmax(dim=-1, keepdim=True)
        return label.data.numpy()

    def _get_avg_embedding(self, lyrics: str) -> torch.Tensor:
        words = word_tokenize(lyrics)
        embedding = np.array([self._word_embedder[word] for word in words])
        avg_embedding = np.mean(embedding, axis=0)
        return torch.from_numpy(avg_embedding)
