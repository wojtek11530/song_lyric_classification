import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from nltk import word_tokenize
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from models.base import BaseModel
from models.fragmentized_lyric_dataset import FragmentizedLyricsDataset
from models.word_embedding.word_embedder import WordEmbedder
from preprocessing.text_preprocessor import fragmentize_text, preprocess, remove_stop_words, lemmatize_text, \
    remove_empty_fragments

_WORKERS_NUM = 1

_PROJECT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TRAIN_DATASET_FILEPATH = os.path.join(_PROJECT_DIRECTORY, 'datasets', 'train_dataset.csv')
_VAL_DATASET_FILEPATH = os.path.join(_PROJECT_DIRECTORY, 'datasets', 'val_dataset.csv')
_TEST_DATASET_FILEPATH = os.path.join(_PROJECT_DIRECTORY, 'datasets', 'test_dataset.csv')


class FragmentizedLSTMClassifier(BaseModel):

    def __init__(self, input_dim: int = 300, learning_rate: float = 1e-3, hidden_dim: int = 100,
                 layer_dim: int = 1, output_dim: int = 4, batch_size: int = 128, weight_decay: float = 1e-5,
                 dropout: float = 0.3, bidirectional: bool = False, max_num_words: Optional[int] = 200,
                 removing_stop_words: bool = False, lemmatization: bool = False):
        super(FragmentizedLSTMClassifier, self).__init__()

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
        self._lemmatization = lemmatization

    def pad_collate(self, batch: List[Tuple[List[np.ndarray], int]]) \
            -> Tuple[List[torch.Tensor], torch.Tensor, List[List[int]]]:
        xx, yy = zip(*batch)
        xx_lens, xx_pad = self._get_padded_embeddings_and_lengths(xx)
        yy = torch.Tensor(yy).to(dtype=torch.int64)
        return xx_pad, yy, xx_lens

    def _get_padded_embeddings_and_lengths(self, xx: Tuple[List[np.ndarray]]) -> Tuple[
        List[List[int]], List[torch.Tensor]]:
        if self._max_num_words:
            xx = [[torch.Tensor(embeddings[:self._max_num_words]) for embeddings in x_fragments]
                  for x_fragments in xx]
        else:
            xx = [[torch.Tensor(embeddings) for embeddings in fragment_embeddings]
                  for fragment_embeddings in xx]
        xx_lens = [[len(fragment_embedding) for fragment_embedding in x_fragments] for x_fragments in xx]
        xx_pad = [pad_sequence(x, batch_first=True, padding_value=0) for x in xx]
        return xx_lens, xx_pad

    def forward(self, xx: List[torch.Tensor], xx_lens: List[List[int]]) -> torch.Tensor:
        output = []
        for x, x_lens in zip(xx, xx_lens):
            fragments_number = len(x_lens)

            x_packed = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

            output_packed, _ = self._lstm(x_packed)
            output_unpacked, _ = pad_packed_sequence(output_packed, batch_first=True)

            seq_len_indices = [length - 1 for length in x_lens]
            fragments_indices = [i for i in range(fragments_number)]
            x_out = output_unpacked[fragments_indices, seq_len_indices, :]
            self._dropout(x_out)
            x_out = self._fc(x_out)

            softmax_out = F.softmax(x_out, dim=1)
            mean_softmax = torch.mean(softmax_out, dim=0, keepdim=True)
            output.append(mean_softmax)

        return torch.cat(output, dim=0)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay)

    def train_dataloader(self) -> DataLoader:
        if self._train_set is None:
            self._train_set = FragmentizedLyricsDataset(_TRAIN_DATASET_FILEPATH,
                                                        removing_stop_words=self._removing_stop_words,
                                                        lemmatization=self._lemmatization)
        return DataLoader(self._train_set, batch_size=self._batch_size, shuffle=False, collate_fn=self.pad_collate)

    def val_dataloader(self) -> DataLoader:
        if self._val_set is None:
            self._val_set = FragmentizedLyricsDataset(_VAL_DATASET_FILEPATH,
                                                      removing_stop_words=self._removing_stop_words,
                                                      lemmatization=self._lemmatization)
        return DataLoader(self._val_set, batch_size=self._batch_size, drop_last=False, collate_fn=self.pad_collate)

    def test_dataloader(self) -> DataLoader:
        if self._test_set is None:
            self._test_set = FragmentizedLyricsDataset(_TEST_DATASET_FILEPATH,
                                                       removing_stop_words=self._removing_stop_words,
                                                       lemmatization=self._lemmatization)
        return DataLoader(self._test_set, batch_size=self._batch_size, drop_last=False, collate_fn=self.pad_collate)

    def training_step(self,
                      batch: Tuple[List[torch.Tensor], torch.Tensor, List[List[int]]],
                      batch_idx: int) -> Dict[str, Any]:
        x, y_labels, x_lens = batch
        softmax_values = self(x, x_lens)
        logits = torch.log(softmax_values)
        loss = F.nll_loss(logits, y_labels)
        total = len(y_labels)
        correct = self._get_correct_prediction_count(softmax_values, y_labels)
        return {'loss': loss, "correct": correct, "total": total, 'log': {'train_loss': loss}}

    def training_epoch_end(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])
        tensorboard_logs = {'loss': avg_loss, "train_acc": correct / total}
        return {'loss': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, val_batch: Tuple[List[torch.Tensor], torch.Tensor, List[List[int]]],
                        batch_idx: int) \
            -> Dict[str, Any]:
        x, y_labels, x_lens = val_batch
        softmax_values = self(x, x_lens)
        logits = torch.log(softmax_values)
        loss = F.nll_loss(logits, y_labels)
        total = len(y_labels)
        correct = self._get_correct_prediction_count(softmax_values, y_labels)
        return {'val_loss': loss, "correct": correct, "total": total}

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) \
            -> Dict[str, Any]:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])

        tensorboard_logs = {'val_loss': avg_loss, "val_acc": correct / total}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def _get_correct_prediction_count(self, probs: torch.Tensor, y_labels: torch.Tensor) -> int:
        return int(probs.argmax(dim=1).eq(y_labels).sum().item())

    def _batch_step(self, batch: List) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y_labels, x_lens = batch
        softmax_values = self(x, x_lens)
        _, y_hat = torch.max(softmax_values, dim=1)
        return y_labels, y_hat

    def predict(self, lyrics: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        lyrics_fragments = fragmentize_text(lyrics)
        lyrics_fragments = [preprocess(fragment, remove_punctuation=True, remove_text_in_brackets=True)
                            for fragment in lyrics_fragments]
        if self._removing_stop_words:
            lyrics_fragments = [remove_stop_words(fragment) for fragment in lyrics_fragments]
        if self._lemmatization:
            lyrics_fragments = [lemmatize_text(fragment) for fragment in lyrics_fragments]
        remove_empty_fragments(lyrics_fragments)

        if not lyrics_fragments:
            return None
        else:
            x, x_lens = self._get_embeddings(lyrics_fragments)
            result = self(x, x_lens)
            probs = torch.squeeze(result)
            label = probs.argmax(dim=-1, keepdim=True)
            return label.data.numpy(), probs.data.numpy()

    def _get_embeddings(self, lyric_fragments: List[str]) -> Tuple[List[torch.Tensor], List[List[int]]]:
        embeddings_of_fragments = []
        for fragment in lyric_fragments:
            words = word_tokenize(fragment)
            embeddings = np.array([self._word_embedder[word] for word in words])
            embeddings_of_fragments.append(embeddings)

        xx_lens, xx_pad = self._get_padded_embeddings_and_lengths((embeddings_of_fragments,))
        return xx_pad, xx_lens
