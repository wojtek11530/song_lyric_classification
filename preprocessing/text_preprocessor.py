import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words('english'))

_TEXT_WITHIN_BRACKETS_REGEX_PATTERN_ = r'[\(\[].*?[\)\]]'


def preprocess(text: str, remove_punctuation: bool, remove_text_in_brackets: bool) -> str:
    text = text.lower()
    if remove_text_in_brackets:
        text = re.sub(_TEXT_WITHIN_BRACKETS_REGEX_PATTERN_, "", text)
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
    return ' '.join([word for word in word_tokenize(text) if word not in STOP_WORDS])
