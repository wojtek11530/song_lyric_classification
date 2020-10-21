import os

import pandas as pd
from sklearn.utils import resample, shuffle

_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATASET_PATH = os.path.join(_PROJECT_PATH, 'datasets')
_DATA_FILE = 'train_dataset.csv'

_RANDOM_SEED = 42

train_df = pd.read_csv(os.path.join(_DATASET_PATH, _DATA_FILE), index_col=0)

happy_df = train_df[train_df['emotion_4Q'] == 'happy']
angry_df = train_df[train_df['emotion_4Q'] == 'angry']
relaxed_df = train_df[train_df['emotion_4Q'] == 'relaxed']
sad_df = train_df[train_df['emotion_4Q'] == 'sad']

happy_downsampled = resample(happy_df,
                             replace=False,
                             n_samples=int((len(angry_df) + len(relaxed_df) + len(sad_df)) / 3),
                             random_state=_RANDOM_SEED)

downsampled_train_df = pd.concat([happy_downsampled, angry_df, relaxed_df, sad_df])
downsampled_train_df = shuffle(downsampled_train_df, random_state=_RANDOM_SEED)
downsampled_train_df.to_csv(os.path.join(_DATASET_PATH, 'downsampled_' + _DATA_FILE))
