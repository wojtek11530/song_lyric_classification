import os

import pandas as pd
from sklearn.model_selection import train_test_split

_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATASET_PATH = os.path.join(_PROJECT_PATH, 'datasets')
_DATA_FILE = 'filtered_dataset_with_lyrics.csv'

_RANDOM_SEED = 42

song_df = pd.read_csv(os.path.join(_DATASET_PATH, _DATA_FILE), index_col=0)

df_train, df_test = train_test_split(
    song_df,
    test_size=0.3,
    random_state=_RANDOM_SEED,
    stratify=song_df['emotion_4Q']
)
df_val, df_test = train_test_split(
    df_test,
    test_size=0.5,
    random_state=_RANDOM_SEED,
    stratify=df_test['emotion_4Q']
)

df_train.to_csv(os.path.join(_DATASET_PATH, 'train_dataset.csv'))
df_val.to_csv(os.path.join(_DATASET_PATH, 'val_dataset.csv'))
df_test.to_csv(os.path.join(_DATASET_PATH, 'test_dataset.csv'))
