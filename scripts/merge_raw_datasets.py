import os
import re

import numpy as np
import pandas as pd

_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATASET_PATH = os.path.join(_PROJECT_PATH, 'datasets')

_COLUMNS_IN_DF_ORDER = ['song_id_from_src',
                        'dataset',
                        'title',
                        'artist',
                        'genre',
                        'arousal_mean',
                        'arousal_std',
                        'valence_mean',
                        'valence_std',
                        'emotion_4Q',
                        'emotion_2Q']


def get_pmemo_data_frame() -> pd.DataFrame:
    pmemo_dataset_name = 'PMEmo2019'

    df_metadata = pd.read_csv(os.path.join(_DATASET_PATH, pmemo_dataset_name, 'metadata.csv'))
    df_metadata = df_metadata.drop(['album', 'fileName', 'duration', 'chorus_start_time', 'chorus_end_time'], axis=1)
    df_annotations_mean = pd.read_csv(os.path.join(_DATASET_PATH, pmemo_dataset_name, 'static_annotations.csv'))
    df_annotations_std = pd.read_csv(os.path.join(_DATASET_PATH, pmemo_dataset_name, 'static_annotations_std.csv'))

    df = df_metadata.merge(df_annotations_mean, on='musicId').merge(df_annotations_std, on='musicId')
    df['dataset'] = pmemo_dataset_name
    df['genre'] = np.nan
    df = df.rename(columns={'musicId': 'song_id_from_src',
                            'Arousal(mean)': 'arousal_mean',
                            'Arousal(std)': 'arousal_std',
                            'Valence(mean)': 'valence_mean',
                            'Valence(std)': 'valence_std'})
    df['emotion_4Q'] = df.apply(lambda x: assess_emotion_four_classes(x['arousal_mean'], x['valence_mean']), axis=1)
    df['emotion_2Q'] = df.apply(lambda x: assess_emotion_two_classes(x['emotion_4Q']), axis=1)
    df = df[_COLUMNS_IN_DF_ORDER]
    return df


def get_emomusic_data_frame() -> pd.DataFrame:
    emomusic_dataset_name = 'emoMusic'

    df_songs = pd.read_csv(os.path.join(_DATASET_PATH, emomusic_dataset_name, 'songs_info.csv'))
    df_annotations = pd.read_csv(os.path.join(_DATASET_PATH, emomusic_dataset_name, 'static_annotations.csv'))

    df_songs = df_songs.drop(
        ['file_name', 'start of the segment (min.sec)', 'end of the segment (min.sec)', 'Mediaeval 2013 set'], axis=1)
    df_songs['Song title'] = df_songs['Song title'].apply(lambda x: re.sub("\t", "", x))
    df_songs['Artist'] = df_songs['Artist'].apply(lambda x: re.sub("\t", "", x))
    df_songs['Genre'] = df_songs['Genre'].apply(lambda x: re.sub("\t", "", x))

    df = df_songs.merge(df_annotations, on='song_id')
    df['dataset'] = emomusic_dataset_name
    df = df.rename(columns={
        'Song title': 'title',
        'Artist': 'artist',
        'Genre': 'genre',
        'mean_arousal': 'arousal_mean',
        'std_arousal': 'arousal_std',
        'mean_valence': 'valence_mean',
        'std_valence': 'valence_std',
        'song_id': 'song_id_from_src'})

    # scaling arousal and valance values from interval [1, 9] into [0, 1]
    df['arousal_mean'] = df.apply(lambda x: (x['arousal_mean'] - 1) / 8, axis=1)
    df['arousal_std'] = df.apply(lambda x: x['arousal_std'] / 8, axis=1)
    df['valence_mean'] = df.apply(lambda x: (x['valence_mean'] - 1) / 8, axis=1)
    df['valence_std'] = df.apply(lambda x: x['valence_std'] / 8, axis=1)

    df['emotion_4Q'] = df.apply(lambda row: assess_emotion_four_classes(row['arousal_mean'], row['valence_mean']),
                                axis=1)
    df['emotion_2Q'] = df.apply(lambda row: assess_emotion_two_classes(row['emotion_4Q']), axis=1)
    df = df[_COLUMNS_IN_DF_ORDER]
    return df


def get_moody_lyrics_data_frame() -> pd.DataFrame:
    moody_lyrics_dataset_name = 'MoodyLyrics4Q'

    df = pd.read_csv(os.path.join(_DATASET_PATH, moody_lyrics_dataset_name, 'MoodyLyrics4Q.csv'))
    df['dataset'] = moody_lyrics_dataset_name
    df['genre'] = np.nan
    df['arousal_mean'] = np.nan
    df['arousal_std'] = np.nan
    df['valence_mean'] = np.nan
    df['valence_std'] = np.nan

    df = df.rename(columns={
        'index': 'song_id_from_src',
        'mood': 'emotion_4Q'})
    df['emotion_2Q'] = df.apply(lambda x: assess_emotion_two_classes(x['emotion_4Q']), axis=1)
    df = df[_COLUMNS_IN_DF_ORDER]
    return df


def assess_emotion_four_classes(arousal: float, valance: float) -> str:
    if arousal >= 0.5 and valance >= 0.5:
        return 'happy'
    elif arousal >= 0.5 and valance < 0.5:
        return 'angry'
    elif arousal < 0.5 and valance >= 0.5:
        return 'relaxed'
    else:
        return 'sad'


def assess_emotion_two_classes(emotion_four_classes: str) -> str:
    if emotion_four_classes == 'happy' or emotion_four_classes == 'relaxed':
        return 'positive'
    else:
        return 'negative'


if __name__ == '__main__':
    pmemo_df = get_pmemo_data_frame()
    emomusic_df = get_emomusic_data_frame()
    get_moody_lyrics_data_frame()
