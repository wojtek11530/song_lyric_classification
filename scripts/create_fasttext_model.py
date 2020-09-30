import os
import re
import shutil

import fasttext
import pandas as pd
from nltk.tokenize import sent_tokenize

from preprocessing.text_preprocessor import lemmatize_text, preprocess, remove_stop_words

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# train_dataset_filepath = os.path.join(project_dir, 'datasets', 'train_dataset.csv')
train_dataset_filepath = os.path.join(project_dir, 'datasets', 'lyrics-data', 'lyrics-data.csv')

temp_lyrics_filename = 'lyrics.txt'
model_filename = 'fasttext_model_200_lemmatization.bin'
model_output = os.path.join(project_dir, 'models', 'word_embedding', 'saved_models', model_filename)

df = pd.read_csv(train_dataset_filepath, index_col=0)
# df = df[df['Idiom'] == 'ENGLISH']


lyrics_data = df.apply(
    lambda x: remove_stop_words(preprocess(x['lyrics'], remove_punctuation=True, remove_text_in_brackets=True)),
    axis=1).values

lyrics_data = df.apply(
    lambda x: lemmatize_text(preprocess(x['lyrics'], remove_punctuation=True, remove_text_in_brackets=True)),
    axis=1).values

# sentences = [re.sub(r'[^\w\s\']', '', sentence) for lyric in lyrics_data for sentence in sent_tokenize(lyric)]
# with open(temp_lyrics_filename, 'w', encoding='utf-8') as f:
#     for sent in sentences:
#         f.write(sent)

with open(temp_lyrics_filename, 'w', encoding='utf-8') as f:
    for lyric in lyrics_data:
        f.write(lyric)

model = fasttext.train_unsupervised(temp_lyrics_filename, dim=200)
model.save_model(model_filename)

shutil.move(model_filename, model_output)

os.remove(temp_lyrics_filename)
