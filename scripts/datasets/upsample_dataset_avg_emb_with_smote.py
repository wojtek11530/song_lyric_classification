import os
from datetime import datetime

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from nltk import word_tokenize

from models.word_embedding.word_embedder import WordEmbedder
from preprocessing.text_preprocessor import lemmatize_text, preprocess, remove_stop_words

_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATASET_PATH = os.path.join(_PROJECT_PATH, 'datasets')
_DATA_FILE = 'train_dataset.csv'

_RANDOM_SEED = 42
_EMBEDDING_DIM = 200

removing_stop_words = True
lemmatization = False

print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Upsampling started")

train_df = pd.read_csv(os.path.join(_DATASET_PATH, _DATA_FILE), index_col=0)
word_embedder = WordEmbedder()

train_df['lyrics'] = train_df.apply(
    lambda x: preprocess(x['lyrics'], remove_punctuation=True, remove_text_in_brackets=True),
    axis=1)

if removing_stop_words:
    train_df['lyrics'] = train_df.apply(lambda x: remove_stop_words(x['lyrics']), axis=1)

if lemmatization:
    train_df['lyrics'] = train_df.apply(lambda x: lemmatize_text(x['lyrics']), axis=1)

train_df = train_df[train_df['lyrics'] != '']
print(train_df['emotion_4Q'].value_counts())

for i in range(_EMBEDDING_DIM):
    train_df['emb_' + str(i)] = None

for line_number, (index, row) in enumerate(train_df.iterrows()):
    text = row['lyrics']
    words = word_tokenize(text)
    embeddings = np.array([word_embedder[word] for word in words])
    avg_embedding = np.mean(embeddings, axis=0)
    for i in range(_EMBEDDING_DIM):
        train_df.at[index, 'emb_' + str(i)] = avg_embedding[i]

train_df = train_df.drop(['arousal_mean', 'arousal_std', 'artist', 'dataset', 'emotion_2Q', 'general_genre', 'genre',
                          'language', 'lyrics', 'song_id_from_src', 'title', 'valence_mean', 'valence_std'], axis=1)

X = np.array(train_df.loc[:, train_df.columns != 'emotion_4Q'])
y = np.array(train_df.loc[:, train_df.columns == 'emotion_4Q'])
print('Shape of X: {}'.format(X.shape))
print('Shape of y: {}'.format(y.shape))

smote = SMOTE(random_state=_RANDOM_SEED)
X_train_res, y_train_res = smote.fit_sample(X, y.ravel())
y_train_res = y_train_res.reshape((-1, 1))

train_res = np.concatenate((y_train_res, X_train_res), axis=1)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

train_df_res = pd.DataFrame(data=train_res,
                            columns=['emotion_4Q'] + ['emb_' + str(i) for i in range(_EMBEDDING_DIM)])

print(train_df_res['emotion_4Q'].value_counts())

train_df_res.to_csv(os.path.join(_DATASET_PATH, 'SMOTE_avg_embedding_' + _DATA_FILE))

print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Upsampling ended")
