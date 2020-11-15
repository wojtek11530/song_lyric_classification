import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from nltk import word_tokenize
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from models.base import BaseModel
from models.fragmentized_lyric_dataset import FragmentizedLyricsDataset
from models.word_embedding.word_embedder import WordEmbedder
from preprocessing.text_preprocessor import fragmentize_text, preprocess, remove_stop_words, lemmatize_text, \
    remove_empty_fragments

_WORKERS_NUM = 4

_PROJECT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TRAIN_DATASET_FILEPATH = os.path.join(_PROJECT_DIRECTORY, 'datasets', 'train_dataset.csv')
_VAL_DATASET_FILEPATH = os.path.join(_PROJECT_DIRECTORY, 'datasets', 'val_dataset.csv')
_TEST_DATASET_FILEPATH = os.path.join(_PROJECT_DIRECTORY, 'datasets', 'test_dataset.csv')


def avg_embedding_collate(batch: List[Tuple[List[np.ndarray], int]]) -> Tuple[List[torch.Tensor], torch.Tensor]:
    xx, yy = zip(*batch)
    xx = [torch.Tensor([np.mean(embeddings, axis=0) for embeddings in fragment_embeddings])
          for fragment_embeddings in xx]
    yy = torch.Tensor(yy).to(dtype=torch.int64)
    return xx, yy


class FragmentizedMLPClassifier(BaseModel):
    def __init__(self, input_size: int = 100, output_size: int = 4,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-5, dropout: float = 0.5, batch_size: int = 128,
                 removing_stop_words: bool = False, lemmatization: bool = False):
        super(FragmentizedMLPClassifier, self).__init__()

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

    def forward(self, xx: List[torch.Tensor]) -> torch.Tensor:
        out = []
        for x in xx:
            x = x.float()
            # layer 1
            x = self._layer_1(x)
            x = F.relu(x)
            x = self._dropout(x)
            # layer 2
            x = self._layer_2(x)

            softmax_out = F.softmax(x, dim=1)
            mean_softmax = torch.mean(softmax_out, dim=0, keepdim=True)

            # x = torch.mean(x, dim=0, keepdim=True)
            out.append(mean_softmax)

        return torch.cat(out, dim=0)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), self._learning_rate, weight_decay=self._weight_decay)

    def train_dataloader(self) -> DataLoader:
        if self._train_set is None:
            self._train_set = FragmentizedLyricsDataset(_TRAIN_DATASET_FILEPATH,
                                                        removing_stop_words=self._removing_stop_words,
                                                        lemmatization=self._lemmatization)
        return DataLoader(self._train_set, batch_size=self._batch_size, shuffle=True,
                          drop_last=False, collate_fn=avg_embedding_collate)

    def val_dataloader(self) -> DataLoader:
        if self._val_set is None:
            self._val_set = FragmentizedLyricsDataset(_VAL_DATASET_FILEPATH,
                                                      removing_stop_words=self._removing_stop_words,
                                                      lemmatization=self._lemmatization)
        return DataLoader(self._val_set, batch_size=self._batch_size, drop_last=False,
                          collate_fn=avg_embedding_collate)

    def test_dataloader(self) -> DataLoader:
        if self._test_set is None:
            self._test_set = FragmentizedLyricsDataset(_TEST_DATASET_FILEPATH,
                                                       removing_stop_words=self._removing_stop_words,
                                                       lemmatization=self._lemmatization)
        return DataLoader(self._test_set, batch_size=self._batch_size, drop_last=False,
                          collate_fn=avg_embedding_collate)

    def training_step(self, batch: Tuple[List[torch.Tensor], torch.Tensor], batch_idx: int) \
            -> Dict[str, Any]:
        x, y_labels = batch
        softmax_values = self(x)
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

    def validation_step(self, val_batch: Tuple[List[torch.Tensor], torch.Tensor], batch_idx: int) \
            -> Dict[str, Any]:
        x, y_labels = val_batch
        softmax_values = self(x)
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
        x, y_labels = batch
        softmax_values = self(x)
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
            avg_embedding = self._get_embeddings(lyrics_fragments)
            result = self(avg_embedding)
            probs = result[0]
            label = probs.argmax(dim=-1, keepdim=True)
            return label.data.numpy(), probs.data.numpy()

    def _get_embeddings(self, lyric_fragments: List[str]) -> List[torch.Tensor]:
        embeddings_of_fragments = []
        for fragment in lyric_fragments:
            words = word_tokenize(fragment)
            embeddings = np.array([self._word_embedder[word] for word in words])
            embeddings_of_fragments.append(embeddings)

        return [torch.Tensor([np.mean(embeddings, axis=0) for embeddings in embeddings_of_fragments])]
