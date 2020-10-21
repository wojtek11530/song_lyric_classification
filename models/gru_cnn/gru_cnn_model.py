import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from nltk import word_tokenize
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from models.base import BaseModel
from models.lyric_dataset import LyricsDataset
from models.word_embedding.word_embedder import WordEmbedder
from preprocessing.text_preprocessor import lemmatize_text, preprocess, remove_stop_words

_WORKERS_NUM = 1

_PROJECT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TRAIN_DATASET_FILEPATH = os.path.join(_PROJECT_DIRECTORY, 'datasets', 'train_dataset.csv')
_VAL_DATASET_FILEPATH = os.path.join(_PROJECT_DIRECTORY, 'datasets', 'val_dataset.csv')
_TEST_DATASET_FILEPATH = os.path.join(_PROJECT_DIRECTORY, 'datasets', 'test_dataset.csv')


class GRUCNNClassifier(BaseModel):

    def __init__(self, input_dim: int = 300, learning_rate: float = 1e-3, gru_hidden_dim: int = 100,
                 gru_layer_dim: int = 1, output_dim: int = 4,
                 filters_number: int = 128, kernels_sizes: Optional[List[int]] = None,
                 batch_size: int = 128, weight_decay: float = 1e-5,
                 dropout: float = 0.3, max_num_words: Optional[int] = 200,
                 removing_stop_words: bool = False, lemmatization: bool = False):
        super(GRUCNNClassifier, self).__init__()

        if kernels_sizes is None:
            kernels_sizes = [3, 5, 7, 9]

        self._train_set: Optional[Dataset] = None
        self._val_set: Optional[Dataset] = None
        self._test_set: Optional[Dataset] = None

        self._embedding_dim = input_dim
        self._gru_hidden_dim = gru_hidden_dim
        self._gru_layer_dim = gru_layer_dim

        self._gru = torch.nn.GRU(input_dim, gru_hidden_dim, gru_layer_dim, batch_first=True, dropout=dropout)

        self._convs = torch.nn.ModuleList([
            torch.nn.Conv1d(in_channels=self._embedding_dim,
                            out_channels=filters_number,
                            kernel_size=kernel_size)
            for kernel_size in kernels_sizes
        ])

        self._dropout = torch.nn.Dropout(p=dropout)
        self._fc = torch.nn.Linear(filters_number * len(kernels_sizes) + self._gru_hidden_dim, output_dim)

        self._learning_rate = learning_rate

        self._batch_size = batch_size
        self._weight_decay = weight_decay

        self._max_num_words = max_num_words
        self._word_embedder = WordEmbedder()
        self._removing_stop_words = removing_stop_words
        self._lemmatization = lemmatization

    def pad_collate(self, batch: List[Tuple[np.ndarray, int]]) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        xx, yy = zip(*batch)
        xx_pad, xx_lens = self._get_padded_embeddings_and_lengths(xx)
        xx_conv_pad = self._get_pad_embeddings_sequence(xx)
        yy = torch.Tensor(yy).to(dtype=torch.int64)
        return xx_pad, xx_conv_pad, yy, xx_lens

    def _get_padded_embeddings_and_lengths(self, xx: Tuple[np.ndarray]) \
            -> Tuple[torch.Tensor, List[int]]:
        if self._max_num_words:
            xx_list = [torch.Tensor(x[:self._max_num_words]) for x in xx]
        else:
            xx_list = [torch.Tensor(x) for x in xx]
        xx_lens = [len(x) for x in xx_list]
        xx_pad = pad_sequence(xx_list, batch_first=True, padding_value=0)
        return xx_pad, xx_lens

    def _get_pad_embeddings_sequence(self, xx: Tuple[np.ndarray]) -> torch.Tensor:
        x_lengths = [len(x) for x in xx]
        xx_pad = np.zeros((len(xx), self._max_num_words, self._embedding_dim))
        for i, x_len in enumerate(x_lengths):
            x = xx[i]
            limit = min(x_len, self._max_num_words)
            xx_pad[i, 0:limit] = x[:limit]
        xx_pad = torch.Tensor(xx_pad)
        return xx_pad

    def forward(self, x: torch.Tensor, x_conv: torch.Tensor, x_lens: List[int]) -> torch.Tensor:

        current_batch_size = len(x_lens)

        x_packed = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

        output_packed, _ = self._gru(x_packed)
        output_unpacked, _ = pad_packed_sequence(output_packed, batch_first=True)

        seq_len_indices = [length - 1 for length in x_lens]
        batch_indices = [i for i in range(current_batch_size)]
        gru_out = output_unpacked[batch_indices, seq_len_indices, :]
        gru_out = F.relu(gru_out)

        x_conv = x_conv.permute(0, 2, 1)
        convolution_layers_outputs = []
        for conv in self._convs:
            conv_out = F.relu(conv(x_conv))
            conv_out = F.max_pool1d(conv_out, kernel_size=conv_out.size()[2])
            convolution_layers_outputs.append(conv_out)

        conv_out = torch.cat(convolution_layers_outputs, 2)
        conv_out = conv_out.reshape(conv_out.size()[0], -1)

        out = torch.cat([gru_out, conv_out], dim=1)

        out = self._dropout(out)
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
                      batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]],
                      batch_idx: int) -> Dict[str, Any]:
        x, x_conv, y_labels, x_lens = batch
        logits = self(x, x_conv, x_lens)
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

    def validation_step(self, val_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]],
                        batch_idx: int) \
            -> Dict[str, Any]:
        x, x_conv, y_labels, x_lens = val_batch
        logits = self(x, x_conv, x_lens)
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
        x, x_conv, y_labels, x_lens = batch
        logits = self(x, x_conv, x_lens)
        _, y_hat = torch.max(logits, dim=1)
        return y_labels, y_hat

    def predict(self, lyrics: str) -> np.ndarray:
        lyrics = preprocess(lyrics, remove_punctuation=True, remove_text_in_brackets=True)
        if self._removing_stop_words:
            lyrics = remove_stop_words(lyrics)
        if self._lemmatization:
            lyrics = lemmatize_text(lyrics)

        padded_embeddings, length = self._get_padded_embeddings_sequence_and_length(lyrics)
        res = torch.squeeze(self(padded_embeddings, length))
        probs = torch.softmax(res, dim=-1)
        label = probs.argmax(dim=-1, keepdim=True)
        return label.data.numpy()

    def _get_padded_embeddings_sequence_and_length(self, lyrics: str) -> Tuple[torch.Tensor, List[int]]:
        words = word_tokenize(lyrics)
        embeddings = (np.array([self._word_embedder[word] for word in words]),)

        xx_pad, xx_lens = self._get_padded_embeddings_and_lengths(embeddings)

        return xx_pad, xx_lens
