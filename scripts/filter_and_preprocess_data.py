import os
import re

import numpy as np
import pandas as pd

_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATASET_PATH = os.path.join(_PROJECT_PATH, 'datasets')
_DATA_FILE = 'dataset_with_lyrics.csv'

_LIMIT = 10000
_LIMIT_WITHOUT_BRACKETS = 5500

_TEXT_WITHIN_BRACKETS_REGEX_PATTERN_ = r'[\(\[].*?[\)\]]'

_GENRE_MAPPER = {
    'Jazz': 'Jazz', 'Soul': 'Jazz', 'Swing': 'Jazz', 'Gospel': 'Jazz', 'Acid Jazz': 'Jazz',
    'Lounge': 'Jazz',
    'Big Band': 'Jazz',

    'Rock': 'Rock', 'Rock & Roll': 'Rock', 'Alternative Rock': 'Rock', 'R&B': 'Rock',
    'Hard Rock': 'Rock',
    'Indie Rock': 'Rock', 'Psychedelic Rock': 'Rock', 'Blues': 'Rock', 'New Wave': 'Rock',
    'Ska': 'Rock',
    'Acoustic': 'Rock', 'Post-Punk': 'Rock', 'Progressive Rock': 'Rock', 'Grunge': 'Rock',
    'HardCore Punk': 'Rock',
    'Pop-Punk': 'Rock', 'Punk Rock': 'Rock', 'Hardcore Punk': 'Rock', 'Classic Rock': 'Rock',
    'Emo': 'Rock', 'Post-Rock': 'Rock', 'Big Beat': 'Rock', 'Goth': 'Rock', 'Stoner Rock': 'Rock',

    'Metal': 'Metal', 'Heavy Metal': 'Metal', 'Thrash Metal': 'Metal', 'Progressive Metal': 'Metal',
    'Symphonic Metal': 'Metal', 'Alternative Metal': 'Metal', 'Doom Metal': 'Metal',
    'Industrial Metal': 'Metal',
    'Metalcore': 'Metal', 'Rap Metal': 'Metal', 'Post-Hardcore': 'Metal', 'Death Metal': 'Metal',
    'Grindcore': 'Metal', 'Gothic Metal': 'Metal', 'Black Metal': 'Metal', 'Nu Metal': 'Metal',
    'Speed Metal': 'Metal',
    'Hardcore': 'Metal', 'Folk Metal': 'Metal',

    'BlueGrass': 'Country', 'Country': 'Country', 'Folk': 'Country',
    'Alternative Country': 'Country',

    'Pop': 'Pop', 'Synthpop': 'Pop', 'Country Pop': 'Pop', 'Indie Pop': 'Pop', 'Indie': 'Pop',
    'Pop-Rock': 'Pop',
    'Reggae': 'Pop', 'Latin': 'Pop', 'Funk': 'Pop', 'Dance': 'Pop', 'Drum & Bass': 'Pop',
    'Euro Dance': 'Pop',
    'Disco': 'Pop',

    'Hip-Hop': 'Hip-Hop', 'Rap': 'Hip-Hop', 'Alternative Hip-Hop': 'Hip-Hop', 'Grime': 'Hip-Hop',
    'Trip Hop': 'Hip-Hop',

    'Electronic': 'Electronic', 'Deep House': 'Electronic', 'House': 'Electronic',
    'Techno': 'Electronic',
    'Electro House': 'Electronic', 'Trance': 'Electronic', 'Breaks': 'Electronic',
    'New Age': 'Electronic',
    'Ambient': 'Electronic', 'Electro-Industrial': 'Electronic', 'UK Garage': 'Electronic',

    'World/Ethnic': None, 'Avant-Garde': None, 'Comedy': None, 'Downtempo': None, 'Classical': None,
    'Experimental': None, 'Singer Songwriter': None,
    None: None, pd.NA: None, np.nan: None
}

song_df = pd.read_csv(os.path.join(_DATASET_PATH, _DATA_FILE), index_col=0)

index_to_drop = [22, 23, 114, 242, 392, 581, 765, 1141, 1144, 1245, 1273, 1342, 1377, 1426, 1822, 1500, 2099, 2309,
                 2537, 2610, 2674, 3159, 3298, 3442]
song_df.drop(index_to_drop, inplace=True)

song_df = song_df[song_df["lyrics"].str.len() < _LIMIT]

for i in song_df.index:
    if not re.search(_TEXT_WITHIN_BRACKETS_REGEX_PATTERN_, song_df.loc[i, "lyrics"]) and \
            len(song_df.loc[i, "lyrics"]) > _LIMIT_WITHOUT_BRACKETS:
        song_df.drop([i], inplace=True)

song_df.drop_duplicates(['lyrics'], inplace=True)
song_df['lyrics'] = song_df.apply(lambda x: x['lyrics'].lower(), axis=1)
song_df['lyrics_with_brackets'] = song_df['lyrics']
song_df['lyrics'] = song_df.apply(
    lambda row: re.sub(_TEXT_WITHIN_BRACKETS_REGEX_PATTERN_, "", row['lyrics']), axis=1)

song_df['lyrics_with_punctuation'] = song_df['lyrics']
song_df['lyrics_with_brackets'] = song_df.apply(lambda x: re.sub(r'[^\w\s\'\[\(\]\)]', '', x['lyrics_with_brackets']),
                                                axis=1)
song_df['lyrics'] = song_df.apply(lambda x: re.sub(r'[^\w\s\']', '', x['lyrics']), axis=1)

for column in ['lyrics', 'lyrics_with_brackets', 'lyrics_with_punctuation']:
    song_df[column] = song_df.apply(lambda x: re.sub(r'\d', '', x[column]), axis=1)
    song_df[column] = song_df.apply(lambda x: re.sub(r'\s+', ' ', x[column]), axis=1)
    song_df[column] = song_df.apply(lambda x: x[column].strip(), axis=1)

song_df = song_df[song_df['lyrics'] != '']

song_df['general_genre'] = song_df.apply(lambda x: _GENRE_MAPPER[x['genre']], axis=1)

song_df.to_csv(os.path.join(_DATASET_PATH, 'preprocessed_' + _DATA_FILE))
