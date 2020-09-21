import os
import re

import pandas as pd

_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATASET_PATH = os.path.join(_PROJECT_PATH, 'datasets')
_DATA_FILE = 'dataset_with_lyrics.csv'

_LIMIT = 10000
_LIMIT_WITHOUT_BRACKETS = 5500

_TEXT_WITH_BRACKETS_REGEX_PATTERN_ = r'[\[\(].*?[\)\]]'

song_df = pd.read_csv(os.path.join(_DATASET_PATH, _DATA_FILE), index_col=0)

index_to_drop = [22, 23, 114, 392, 1141, 1144, 1245, 1273, 1342, 1426, 1500, 2099, 2309, 2537, 1377, 3159, 3298, 3442]
song_df.drop(index_to_drop, inplace=True)

song_df = song_df[song_df["lyrics"].str.len() < _LIMIT]

for i in song_df.index:
    if not re.search(_TEXT_WITH_BRACKETS_REGEX_PATTERN_, song_df.loc[i, "lyrics"]) and \
            len(song_df.loc[i, "lyrics"]) > _LIMIT_WITHOUT_BRACKETS:
        song_df.drop([i], inplace=True)

song_df['lyrics'] = song_df.apply(lambda x: re.sub(r'\s+', ' ', x['lyrics']), axis=1)
song_df['lyrics'] = song_df.apply(lambda x: x['lyrics'].lstrip(), axis=1)

song_df['lyrics_without_brackets'] = song_df.apply(
    lambda row: re.sub(_TEXT_WITH_BRACKETS_REGEX_PATTERN_, "", row['lyrics']), axis=1)

song_df['lyrics_without_brackets'] = song_df.apply(lambda x: re.sub(r'\s+', ' ', x['lyrics_without_brackets']), axis=1)
song_df['lyrics_without_brackets'] = song_df.apply(lambda x: x['lyrics_without_brackets'].lstrip(), axis=1)

song_df.to_csv(os.path.join(_DATASET_PATH, 'preprocessed_' + _DATA_FILE))
