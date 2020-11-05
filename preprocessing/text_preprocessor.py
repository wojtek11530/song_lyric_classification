import re
from typing import Any, List

import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

wordnet_lemmatizer = WordNetLemmatizer()

STOP_WORDS = set(stopwords.words('english'))

NOT_USED_STOP_WORDS = {'more', 'aren', "mightn't", 'doesn', 'isn', "didn't", 'wouldn', "won't", 'ain', 'couldn',
                       "shouldn't", "weren't", 'didn', "hadn't", 'needn', 'shouldn', 'mustn', "mustn't", "wasn't",
                       "couldn't", 'wasn', "hasn't", 'very', 'most', 'hadn', "wouldn't", "don't", "aren't", 'hasn',
                       "needn't", "haven't", 'nor', 'no', 'won', 'not', 'haven', "isn't", 'don', "doesn't"}

ADDITIONAL_STOP_WORDS = {"'s", "'re", "'m", "'ve", "'d", "'ll"}

STOP_WORDS = STOP_WORDS - NOT_USED_STOP_WORDS | ADDITIONAL_STOP_WORDS

_TEXT_WITHIN_BRACKETS_REGEX_PATTERN = r'\[.*?\]'

_WORDS_IN_FRAGMENT = 60


def preprocess(text: str, remove_punctuation: bool, remove_text_in_brackets: bool, expand_contraction: bool = False) \
        -> str:
    text = text.lower()
    if expand_contraction:
        text = contractions.fix(text)
    if remove_text_in_brackets:
        text = re.sub(_TEXT_WITHIN_BRACKETS_REGEX_PATTERN, "", text)
    if remove_punctuation:
        if remove_text_in_brackets:
            text = re.sub(r'[^\w\s\']', '', text)
        else:
            re.sub(r'[^\w\s\'\[(\])]', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def remove_stop_words(text: str) -> str:
    text_without_stop_words = ' '.join([word for word in word_tokenize(text) if word not in STOP_WORDS])
    text_without_stop_words = re.sub(r'\s+\'\s+', ' ', text_without_stop_words)
    return text_without_stop_words


def lemmatize_text(text: str) -> str:
    return ' '.join([wordnet_lemmatizer.lemmatize(word) for word in word_tokenize(text)])


def fragmentize_text(text: str) -> List[str]:
    if re.search(_TEXT_WITHIN_BRACKETS_REGEX_PATTERN, text):
        fragments = _get_fragments_split_by_square_brackets(text)
    else:
        fragments = _get_fragments_with_even_number_of_words(text, _WORDS_IN_FRAGMENT)

    return fragments


def _get_fragments_split_by_square_brackets(text: str) -> List[str]:
    fragments = re.split(_TEXT_WITHIN_BRACKETS_REGEX_PATTERN, text)
    remove_empty_fragments(fragments)
    return fragments


def remove_empty_fragments(fragments: List[str]) -> None:
    index_to_delete = []
    for i in range(len(fragments)):
        text_fragment = fragments[i]
        text_fragment = re.sub(r'\s+', ' ', text_fragment)
        fragments[i] = text_fragment
        if fragments[i] in ['', ' ']:
            index_to_delete.append(i)
    for index in sorted(index_to_delete, reverse=True):
        del fragments[index]


def _get_fragments_with_even_number_of_words(text: str, words_number: int) -> List[str]:
    words = word_tokenize(text)
    words_in_fragments = _get_chunks(words, words_number)
    fragments = [' '.join(words) for words in words_in_fragments]
    return fragments


def _get_chunks(lst: List[Any], chunk_number: int) -> List[List[Any]]:
    chunks = []
    for i in range(0, len(lst), chunk_number):
        chunks.append(lst[i:i + chunk_number])
    return chunks
