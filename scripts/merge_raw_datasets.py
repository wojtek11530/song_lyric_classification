import os

import pandas as pd

_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATASET_PATH = os.path.join(_PROJECT_PATH, 'datasets')

_COLUMNS_IN_DF_ORDER = ['song_id_from_src',
                        'dataset',
                        'title',
                        'artist',
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
    df = df.rename(columns={'musicId': 'song_id_from_src',
                            'Arousal(mean)': 'arousal_mean',
                            'Arousal(std)': 'arousal_std',
                            'Valence(mean)': 'valence_mean',
                            'Valence(std)': 'valence_std'})
    df['emotion_4Q'] = df.apply(lambda x: assess_emotion_four_classes(x['arousal_mean'], x['valence_mean']), axis=1)
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
