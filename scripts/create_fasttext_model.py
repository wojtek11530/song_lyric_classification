import os
# import re
import shutil

import fasttext
import pandas as pd
# from nltk.tokenize import sent_tokenize

from preprocessing.text_preprocessor import lemmatize_text, preprocess, remove_stop_words

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_LYRICS_FILENAME = 'lyrics.txt'


def run_creating_fasttext_model() -> None:
    create_fasttext_model(
        dim=200,
        large_dataset=True,
        remove_stopwords=True,
        lemmatization=False,
    )


def create_fasttext_model(large_dataset: bool, remove_stopwords: bool, lemmatization: bool, dim: int = 200) -> None:
    model_filename = 'fasttext_model_' + str(dim)

    if large_dataset:
        train_dataset_filepath = os.path.join(PROJECT_DIR, 'datasets', 'lyrics-data', 'lyrics-data.csv')
        df = pd.read_csv(train_dataset_filepath, index_col=0)
        df = df[df['Idiom'] == 'ENGLISH']
        lyric_column_name = 'Lyric'
        model_filename += '_large'
    else:
        train_dataset_filepath = os.path.join(PROJECT_DIR, 'datasets', 'train_dataset.csv')
        df = pd.read_csv(train_dataset_filepath, index_col=0)
        lyric_column_name = 'lyrics'

    df[lyric_column_name] = df.apply(lambda x: preprocess(x[lyric_column_name],
                                                          remove_punctuation=True,
                                                          remove_text_in_brackets=True),
                                     axis=1)
    if remove_stopwords:
        df[lyric_column_name] = df.apply(lambda x: remove_stop_words(x[lyric_column_name]), axis=1)
        model_filename += '_stopwords_removed'

    if lemmatization:
        df[lyric_column_name] = df.apply(lambda x: lemmatize_text(x[lyric_column_name]), axis=1)
        model_filename += '_lemmatization'

    model_filename += '.bin'
    lyrics_data = df[lyric_column_name].values

    # lyrics_data = df.apply(
    #     lambda x: lemmatize_text(preprocess(x['Lyric'], remove_punctuation=True, remove_text_in_brackets=True)),
    #     axis=1).values
    # sentences = [re.sub(r'[^\w\s\']', '', sentence) for lyric in lyrics_data for sentence in sent_tokenize(lyric)]
    # with open(temp_lyrics_filename, 'w', encoding='utf-8') as f:
    #     for sent in sentences:
    #         f.write(sent)

    with open(TEMP_LYRICS_FILENAME, 'w', encoding='utf-8') as f:
        for lyric in lyrics_data:
            f.write(lyric)

    model = fasttext.train_unsupervised(TEMP_LYRICS_FILENAME, dim=dim)
    model_output = os.path.join(PROJECT_DIR, 'models', 'word_embedding', 'saved_models', model_filename)
    model.save_model(model_filename)
    shutil.move(model_filename, model_output)
    os.remove(TEMP_LYRICS_FILENAME)


if __name__ == '__main__':
    run_creating_fasttext_model()
