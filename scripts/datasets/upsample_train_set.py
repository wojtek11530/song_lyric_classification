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

angry_upsampled = resample(angry_df,
                           replace=True,
                           n_samples=len(happy_df),
                           random_state=_RANDOM_SEED)

relaxed_upsampled = resample(relaxed_df,
                             replace=True,
                             n_samples=len(happy_df),
                             random_state=_RANDOM_SEED)

sad_upsampled = resample(sad_df,
                         replace=True,
                         n_samples=len(happy_df),
                         random_state=_RANDOM_SEED)

upsampled_train_df = pd.concat([happy_df, angry_upsampled, relaxed_upsampled, sad_upsampled])
upsampled_train_df = shuffle(upsampled_train_df, random_state=_RANDOM_SEED)
upsampled_train_df.to_csv(os.path.join(_DATASET_PATH, 'upsampled_' + _DATA_FILE))
