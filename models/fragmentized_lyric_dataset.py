from typing import List, Tuple

import numpy as np
import pandas as pd
from nltk import word_tokenize
from torch.utils.data import Dataset

from models.label_encoder import label_encoder
from models.word_embedding.word_embedder import WordEmbedder
from preprocessing.text_preprocessor import (
    fragmentize_text, lemmatize_text, preprocess, remove_empty_fragments, remove_stop_words)


class FragmentizedLyricsDataset(Dataset):
    def __init__(self, filepath: str, removing_stop_words: bool = False, lemmatization: bool = False):
        song_df = pd.read_csv(filepath, index_col=0)

        self.emotion_data, self.lyrics_data = self._get_fragmentized_lyrics_and_emotion_data(
            song_df,
            removing_stop_words,
            lemmatization
        )
        self.word_embedder = WordEmbedder()

    def __getitem__(self, index: int) -> Tuple[List[np.ndarray], int]:
        return self._get_embeddings(self.lyrics_data[index]), \
               self.emotion_data[index]

    def __len__(self) -> int:
        return len(self.emotion_data)

    def _get_embeddings(self, lyric_fragments: List[str]) -> List[np.ndarray]:
        embeddings_of_fragments = []
        for fragment in lyric_fragments:
            words = word_tokenize(fragment)
            embeddings = np.array([self.word_embedder[word] for word in words])
            embeddings_of_fragments.append(embeddings)
        return embeddings_of_fragments

    @staticmethod
    def _get_fragmentized_lyrics_and_emotion_data(song_df: pd.DataFrame, removing_stop_words: bool,
                                                  lemmatization: bool) -> Tuple[np.ndarray, List[List[str]]]:
        emotion_labels = list(song_df['emotion_4Q'].values)
        lyrics_data = song_df['lyrics'].values
        lyrics_data = [fragmentize_text(lyrics) for lyrics in lyrics_data]
        lyrics_data = [[preprocess(fragment, remove_punctuation=True, remove_text_in_brackets=True)
                        for fragment in fragments] for fragments in lyrics_data]
        if removing_stop_words:
            lyrics_data = [[remove_stop_words(fragment) for fragment in fragments] for fragments in lyrics_data]
        if lemmatization:
            lyrics_data = [[lemmatize_text(fragment) for fragment in fragments] for fragments in lyrics_data]
        for fragments in lyrics_data:
            remove_empty_fragments(fragments)
        FragmentizedLyricsDataset._remove_records_without_fragments(emotion_labels, lyrics_data)

        emotion_data = label_encoder.transform(emotion_labels)
        return emotion_data, lyrics_data

    @staticmethod
    def _remove_records_without_fragments(emotion_labels: List[str], lyrics_data: List[List[str]]):
        index_to_delete = []
        for i in range(len(lyrics_data)):
            fragments = lyrics_data[i]
            if len(fragments) == 0:
                index_to_delete.append(i)
        for index in sorted(index_to_delete, reverse=True):
            del lyrics_data[index]
            del emotion_labels[index]
