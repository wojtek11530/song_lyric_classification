import os
import re

import numpy as np
import pandas as pd

_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

    pmemo_df = df_metadata.merge(df_annotations_mean, on='musicId').merge(df_annotations_std, on='musicId')
    pmemo_df['dataset'] = pmemo_dataset_name
    pmemo_df['genre'] = np.nan
    pmemo_df = pmemo_df.rename(columns={'musicId': 'song_id_from_src',
                                        'Arousal(mean)': 'arousal_mean',
                                        'Arousal(std)': 'arousal_std',
                                        'Valence(mean)': 'valence_mean',
                                        'Valence(std)': 'valence_std'})
    pmemo_df['emotion_4Q'] = pmemo_df.apply(lambda x: assess_emotion_four_classes(x['arousal_mean'], x['valence_mean']),
                                            axis=1)
    pmemo_df['emotion_2Q'] = pmemo_df.apply(lambda x: assess_emotion_two_classes(x['emotion_4Q']), axis=1)
    pmemo_df = pmemo_df[_COLUMNS_IN_DF_ORDER]
    return pmemo_df


def get_emomusic_data_frame() -> pd.DataFrame:
    emomusic_dataset_name = 'emoMusic'

    df_songs = pd.read_csv(os.path.join(_DATASET_PATH, emomusic_dataset_name, 'songs_info.csv'))
    df_annotations = pd.read_csv(os.path.join(_DATASET_PATH, emomusic_dataset_name, 'static_annotations.csv'))

    df_songs = df_songs.drop(
        ['file_name', 'start of the segment (min.sec)', 'end of the segment (min.sec)', 'Mediaeval 2013 set'], axis=1)
    df_songs['Song title'] = df_songs['Song title'].apply(lambda x: re.sub("\t", "", x))
    df_songs['Artist'] = df_songs['Artist'].apply(lambda x: re.sub("\t", "", x))
    df_songs['Genre'] = df_songs['Genre'].apply(lambda x: re.sub("\t", "", x))

    emomusic_df = df_songs.merge(df_annotations, on='song_id')
    emomusic_df['dataset'] = emomusic_dataset_name
    emomusic_df = emomusic_df.rename(columns={
        'Song title': 'title',
        'Artist': 'artist',
        'Genre': 'genre',
        'mean_arousal': 'arousal_mean',
        'std_arousal': 'arousal_std',
        'mean_valence': 'valence_mean',
        'std_valence': 'valence_std',
        'song_id': 'song_id_from_src'})

    # scaling arousal and valance values from interval [1, 9] into [0, 1]
    emomusic_df['arousal_mean'] = emomusic_df.apply(lambda x: (x['arousal_mean'] - 1) / 8, axis=1)
    emomusic_df['arousal_std'] = emomusic_df.apply(lambda x: x['arousal_std'] / 8, axis=1)
    emomusic_df['valence_mean'] = emomusic_df.apply(lambda x: (x['valence_mean'] - 1) / 8, axis=1)
    emomusic_df['valence_std'] = emomusic_df.apply(lambda x: x['valence_std'] / 8, axis=1)

    emomusic_df['emotion_4Q'] = emomusic_df.apply(
        lambda row: assess_emotion_four_classes(row['arousal_mean'], row['valence_mean']),
        axis=1)
    emomusic_df['emotion_2Q'] = emomusic_df.apply(lambda row: assess_emotion_two_classes(row['emotion_4Q']), axis=1)
    emomusic_df = emomusic_df[_COLUMNS_IN_DF_ORDER]
    return emomusic_df


def get_moody_lyrics_data_frame() -> pd.DataFrame:
    moody_lyrics_dataset_name = 'MoodyLyrics4Q'

    moody_lyrics_df = pd.read_csv(os.path.join(_DATASET_PATH, moody_lyrics_dataset_name, 'MoodyLyrics4Q.csv'))
    moody_lyrics_df['dataset'] = moody_lyrics_dataset_name
    moody_lyrics_df['genre'] = np.nan
    moody_lyrics_df['arousal_mean'] = np.nan
    moody_lyrics_df['arousal_std'] = np.nan
    moody_lyrics_df['valence_mean'] = np.nan
    moody_lyrics_df['valence_std'] = np.nan

    moody_lyrics_df = moody_lyrics_df.rename(columns={
        'index': 'song_id_from_src',
        'mood': 'emotion_4Q'})
    moody_lyrics_df['emotion_2Q'] = moody_lyrics_df.apply(lambda x: assess_emotion_two_classes(x['emotion_4Q']), axis=1)
    moody_lyrics_df = moody_lyrics_df[_COLUMNS_IN_DF_ORDER]
    return moody_lyrics_df


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
    moodyLyrics4Q_df = get_moody_lyrics_data_frame()

    df = pd.concat([pmemo_df, emomusic_df, moodyLyrics4Q_df], ignore_index=True)
    merged_dataset_filename = 'merged_datasets.csv'
    df.to_csv(os.path.join(_DATASET_PATH, merged_dataset_filename))
