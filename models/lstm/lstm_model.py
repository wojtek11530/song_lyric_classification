import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from models.base import BaseModel
from models.lyric_dataset import LyricsDataset
from models.word_embedding.word_embedder import WordEmbedder

_WORKERS_NUM = 1

_PROJECT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TRAIN_DATASET_FILEPATH = os.path.join(_PROJECT_DIRECTORY, 'datasets', 'train_dataset.csv')
_VAL_DATASET_FILEPATH = os.path.join(_PROJECT_DIRECTORY, 'datasets', 'val_dataset.csv')
_TEST_DATASET_FILEPATH = os.path.join(_PROJECT_DIRECTORY, 'datasets', 'test_dataset.csv')


class LSTMClassifier(BaseModel):

    def __init__(self, input_dim: int = 300, learning_rate: float = 1e-3, hidden_dim: int = 100,
                 layer_dim: int = 1, output_dim: int = 4, batch_size: int = 128, weight_decay: float = 1e-5,
                 dropout: float = 0.3, bidirectional: bool = False, max_num_words: Optional[int] = 200,
                 removing_stop_words: bool = False):
        super(LSTMClassifier, self).__init__()

        self._train_set: Optional[Dataset] = None
        self._val_set: Optional[Dataset] = None
        self._test_set: Optional[Dataset] = None

        self._hidden_dim = hidden_dim
        self._layer_dim = layer_dim
        self._num_directions = 2 if bidirectional else 1

        self._lstm = torch.nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout,
                                   bidirectional=bidirectional)
        self._fc = torch.nn.Linear(hidden_dim * self._num_directions, output_dim)
        self._dropout = torch.nn.Dropout(p=dropout)

        self._learning_rate = learning_rate

        self._batch_size = batch_size
        self._weight_decay = weight_decay

        self._max_num_words = max_num_words
        self._word_embedder = WordEmbedder()
        self._removing_stop_words = removing_stop_words

    def pad_collate(self, batch: List[Tuple[np.ndarray, int]]) \
            -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        xx, yy = zip(*batch)
        if self._max_num_words:
            xx = [torch.Tensor(x[:self._max_num_words]) for x in xx]
        else:
            xx = [torch.Tensor(x) for x in xx]
        yy = torch.Tensor(yy).to(dtype=torch.int64)
        xx_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
        return xx_pad, yy, xx_lens

    def forward(self, x: torch.Tensor, x_lens: List[int]) -> torch.Tensor:

        current_batch_size = len(x_lens)

        x_packed = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

        output_packed, _ = self._lstm(x_packed)
        output_unpacked, _ = pad_packed_sequence(output_packed, batch_first=True)

        seq_len_indices = [length - 1 for length in x_lens]
        batch_indices = [i for i in range(current_batch_size)]
        out = output_unpacked[batch_indices, seq_len_indices, :]
        self._dropout(out)
        out = self._fc(out)
        return out

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay)

    def train_dataloader(self) -> DataLoader:
        if self._train_set is None:
            self._train_set = LyricsDataset(_TRAIN_DATASET_FILEPATH, self._removing_stop_words)
        return DataLoader(self._train_set, batch_size=self._batch_size, shuffle=False, collate_fn=self.pad_collate)

    def val_dataloader(self) -> DataLoader:
        if self._val_set is None:
            self._val_set = LyricsDataset(_VAL_DATASET_FILEPATH, self._removing_stop_words)
        return DataLoader(self._val_set, batch_size=self._batch_size, drop_last=False, collate_fn=self.pad_collate)

    def test_dataloader(self) -> DataLoader:
        if self._test_set is None:
            self._test_set = LyricsDataset(_TEST_DATASET_FILEPATH, self._removing_stop_words)
        return DataLoader(self._test_set, batch_size=self._batch_size, drop_last=False, collate_fn=self.pad_collate)

    def training_step(self,
                      batch: Tuple[torch.Tensor, torch.Tensor, List[int]],
                      batch_idx: int) -> Dict[str, Any]:
        x, y_labels, x_lens = batch
        logits = self(x, x_lens)
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

    def validation_step(self, val_batch: Tuple[torch.Tensor, torch.Tensor, List[int]],
                        batch_idx: int) \
            -> Dict[str, Any]:
        x, y_labels, x_lens = val_batch
        logits = self(x, x_lens)
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

    def _batch_step(self, batch: List) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y_labels, x_lens = batch
        logits = self(x, x_lens)
        _, y_hat = torch.max(logits, dim=1)
        return y_labels, y_hat
