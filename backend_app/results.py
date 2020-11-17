import os
from typing import Dict, Optional

import pandas as pd

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_FILENAME = 'results.csv'
_PATH = os.path.join(_BACKEND_DIR, _FILENAME)


def add_to_results(title: str, artist: str, results: Dict[str, float]):
    song_data_dict = {'title': title, 'artist': artist}
    song_data_dict.update(results)

    if os.path.isfile(_PATH):
        df = pd.read_csv(_PATH)
        new_df = pd.DataFrame({k: [v] for k, v in song_data_dict.items()})
        df = df.append(new_df)
    else:
        df = pd.DataFrame({k: [v] for k, v in song_data_dict.items()})

    df.to_csv(_PATH, index=False)


def get_average_results() -> Optional[Dict[str, float]]:
    if os.path.isfile(_PATH):
        df = pd.read_csv(_PATH)
        df.drop(columns=['title', 'artist'], inplace=True)
        average_emotions = df.mean(axis=0)
        emotion_probabilities = {}
        for emotion, value in average_emotions.iteritems():
            emotion_probabilities[emotion] = value
        return emotion_probabilities
    else:
        return None
