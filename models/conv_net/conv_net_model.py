import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from models.base import BaseModel
from models.lyric_dataset import LyricsDataset
from models.word_embedding.word_embedder import WordEmbedder

_WORKERS_NUM = 1

_PROJECT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TRAIN_DATASET_FILEPATH = os.path.join(_PROJECT_DIRECTORY, 'datasets', 'train_dataset.csv')
_VAL_DATASET_FILEPATH = os.path.join(_PROJECT_DIRECTORY, 'datasets', 'val_dataset.csv')
_TEST_DATASET_FILEPATH = os.path.join(_PROJECT_DIRECTORY, 'datasets', 'test_dataset.csv')


class ConvNetClassifier(BaseModel):

    def __init__(self, embedding_dim: int = 300, output_dim: int = 4, batch_size: int = 128, dropout: float = 0.3,
                 learning_rate: float = 1e-3, filters_number: int = 128, kernels_sizes: Optional[List[int]] = None,
                 weight_decay: float = 1e-5, max_num_words: int = 200,
                 removing_stop_words: bool = False, lemmatization: bool = False):
        super(ConvNetClassifier, self).__init__()

        if kernels_sizes is None:
            kernels_sizes = [3, 5, 7, 9]

        self._train_set: Optional[Dataset] = None
        self._val_set: Optional[Dataset] = None
        self._test_set: Optional[Dataset] = None

        self._embedding_dim = embedding_dim
        self._max_num_words = max_num_words
        self._word_embedder = WordEmbedder()
        self._removing_stop_words = removing_stop_words
        self._lemmatization = lemmatization

        self._convs = torch.nn.ModuleList([
            torch.nn.Conv1d(in_channels=self._embedding_dim,
                            out_channels=filters_number,
                            kernel_size=kernel_size)
            for kernel_size in kernels_sizes
        ])
        self._fc = torch.nn.Linear(filters_number * len(kernels_sizes), output_dim)
        self._dropout = torch.nn.Dropout(p=dropout)

        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._weight_decay = weight_decay

    def pad_collate(self, batch: List[Tuple[np.ndarray, int]]) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        xx, yy = zip(*batch)

        x_lengths = [len(x) for x in xx]
        xx_pad = np.zeros((len(xx), self._max_num_words, self._embedding_dim))

        for i, x_len in enumerate(x_lengths):
            x = xx[i]
            limit = min(x_len, self._max_num_words)
            xx_pad[i, 0:limit] = x[:limit]
        xx_pad = torch.Tensor(xx_pad)
        yy = torch.Tensor(yy).to(dtype=torch.int64)

        return xx_pad, yy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        convolution_layers_outputs = []
        for conv in self._convs:
            conv_out = F.relu(conv(x))
            conv_out = conv_out.squeeze()
            conv_out = F.max_pool1d(conv_out, kernel_size=conv_out.size()[2])
            convolution_layers_outputs.append(conv_out)

        out = torch.cat(convolution_layers_outputs, 2)
        out = out.reshape(out.size()[0], -1)
        # out = self._dropout(out)
        out = self._fc(out)
        return out

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay)

    def train_dataloader(self) -> DataLoader:
        if self._train_set is None:
            self._train_set = LyricsDataset(_TRAIN_DATASET_FILEPATH,
                                            removing_stop_words=self._removing_stop_words,
                                            lemmatization=self._lemmatization)
        return DataLoader(self._train_set, batch_size=self._batch_size, shuffle=False, collate_fn=self.pad_collate)

    def val_dataloader(self) -> DataLoader:
        if self._val_set is None:
            self._val_set = LyricsDataset(_VAL_DATASET_FILEPATH,
                                          removing_stop_words=self._removing_stop_words,
                                          lemmatization=self._lemmatization)
        return DataLoader(self._val_set, batch_size=self._batch_size, drop_last=False, collate_fn=self.pad_collate)

    def test_dataloader(self) -> DataLoader:
        if self._test_set is None:
            self._test_set = LyricsDataset(_TEST_DATASET_FILEPATH,
                                           removing_stop_words=self._removing_stop_words,
                                           lemmatization=self._lemmatization)
        return DataLoader(self._test_set, batch_size=self._batch_size, drop_last=False, collate_fn=self.pad_collate)

    def training_step(self,
                      batch: Tuple[torch.nn.utils.rnn.PackedSequence, torch.Tensor],
                      batch_idx: int) -> Dict[str, Any]:
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

    def validation_step(self, val_batch: Tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int) \
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

    def predict(self, sentence: str) -> np.ndarray:
        embeddings = self._get_embeddings(sentence)
        res = torch.squeeze(self(embeddings))
        probs = torch.softmax(res, dim=-1)
        label = probs.argmax(dim=-1, keepdim=True)
        return label.data.numpy()

    def _get_embeddings(self, sentence: str) -> List[torch.Tensor]:
        words = sentence.split()
        embeddings = [self._word_embedder[word] for word in words]
        embeddings = [torch.Tensor(embeddings)]
        return embeddings

    def _batch_step(self, batch: List) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y_labels = batch
        logits = self(x)
        _, y_hat = torch.max(logits, dim=1)
        return y_labels, y_hat
