import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from models.base import BaseModel
from models.lyric_dataset import LyricsDataset
from models.word_embedding.word_embedder import WordEmbedder

_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_WORKERS_NUM = 4

_PROJECT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TRAIN_DATASET_FILEPATH = os.path.join(_PROJECT_DIRECTORY, 'datasets', 'train_dataset.csv')
_VAL_DATASET_FILEPATH = os.path.join(_PROJECT_DIRECTORY, 'datasets', 'val_dataset.csv')
_TEST_DATASET_FILEPATH = os.path.join(_PROJECT_DIRECTORY, 'datasets', 'test_dataset.csv')


def pad_collate(batch: List[Tuple[np.ndarray, int]]) \
        -> Tuple[torch.nn.utils.rnn.PackedSequence, torch.Tensor]:
    xx, yy = zip(*batch)
    xx = [torch.Tensor(x) for x in xx]
    yy = torch.Tensor(yy).to(dtype=torch.int64)
    xx_lens = [len(x) for x in xx]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)

    xx_packed = pack_padded_sequence(xx_pad, xx_lens, batch_first=True, enforce_sorted=False)
    return xx_packed, yy


class LSTMClassifier(BaseModel):

    def __init__(self, input_dim: int = 300, learning_rate: float = 1e-3, hidden_dim: int = 100,
                 layer_dim: int = 1, output_dim: int = 4, batch_size: int = 128,
                 dropout: float = 0.3, bidirectional: bool = False):
        super(LSTMClassifier, self).__init__()

        self._train_set: Optional[Dataset] = None
        self._val_set: Optional[Dataset] = None
        self._test_set: Optional[Dataset] = None

        self._hidden_dim = hidden_dim
        self._layer_dim = layer_dim
        self._num_directions = 2 if bidirectional else 1

        self._lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout,
                             bidirectional=bidirectional)

        self._fc = nn.Linear(hidden_dim * self._num_directions, output_dim)
        self._learning_rate = learning_rate
        self._batch_size = batch_size

        self._word_embedder = WordEmbedder()

        self.to(_DEVICE)

    def forward(self, x: torch.nn.utils.rnn.PackedSequence) -> torch.Tensor:

        current_batch_size = x.batch_sizes[0]

        h0 = torch.zeros(self._layer_dim * self._num_directions, current_batch_size,
                         self._hidden_dim)

        c0 = torch.zeros(self._layer_dim * self._num_directions, current_batch_size,
                         self._hidden_dim)

        output_packed, (hn, cn) = self._lstm(x, (h0, c0))
        output_unpacked, output_lengths = pad_packed_sequence(output_packed, batch_first=True)

        seq_len_indices = [length - 1 for length in output_lengths]
        batch_indices = [i for i in range(current_batch_size)]
        out = output_unpacked[batch_indices, seq_len_indices, :]

        out = self._fc(out)
        return out

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), self._learning_rate)

    def train_dataloader(self) -> DataLoader:
        if self._train_set is None:
            self._train_set = LyricsDataset(_TRAIN_DATASET_FILEPATH)
        return DataLoader(self._train_set, batch_size=self._batch_size, shuffle=True,
                          drop_last=True, collate_fn=pad_collate, num_workers=_WORKERS_NUM)

    def val_dataloader(self) -> DataLoader:
        if self._val_set is None:
            self._val_set = LyricsDataset(_VAL_DATASET_FILEPATH)
        return DataLoader(self._val_set, batch_size=self._batch_size, drop_last=True,
                          collate_fn=pad_collate, num_workers=_WORKERS_NUM)

    def test_dataloader(self) -> DataLoader:
        if self._test_set is None:
            self._test_set = LyricsDataset(_TEST_DATASET_FILEPATH)
        return DataLoader(self._test_set, batch_size=self._batch_size, drop_last=True,
                          collate_fn=pad_collate, num_workers=_WORKERS_NUM)

    def training_step(self,
                      batch: Tuple[torch.nn.utils.rnn.PackedSequence, torch.Tensor],
                      batch_idx: int) -> Dict[str, Any]:
        x, y_labels = batch
        x = x.to(_DEVICE)
        y_labels = y_labels.to(_DEVICE)
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

    def validation_step(self, val_batch: Tuple[torch.nn.utils.rnn.PackedSequence, torch.Tensor],
                        batch_idx: int) \
            -> Dict[str, Any]:
        x, y_labels = val_batch
        x = x.to(_DEVICE)
        y_labels = y_labels.to(_DEVICE)
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

    def predict(self, sentence: str) -> np.ndarray:
        embeddings = self._get_embeddings(sentence)
        res = torch.squeeze(self(embeddings))
        probs = torch.softmax(res, dim=-1)
        label = probs.argmax(dim=-1, keepdim=True)
        return label.data.numpy()

    def _get_embeddings(self, sentence: str) -> torch.nn.utils.rnn.PackedSequence:
        words = sentence.split()
        embeddings = [self._word_embedder[word] for word in words]
        embeddings = [torch.Tensor(embeddings)]

        embeddings_pad = pad_sequence(embeddings, batch_first=True, padding_value=0)
        embeddings_packed = pack_padded_sequence(embeddings_pad, [len(words)], batch_first=True,
                                                 enforce_sorted=False)
        return embeddings_packed
