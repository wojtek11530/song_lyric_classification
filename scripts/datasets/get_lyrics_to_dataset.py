import os
import re
from datetime import datetime
from typing import Optional

import lyricsgenius
import numpy as np
import pandas as pd
import requests
from langdetect import detect
from requests.exceptions import HTTPError

_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATASET_PATH = os.path.join(_PROJECT_PATH, 'datasets')
_DATA_FILE = 'merged_datasets.csv'
_NEW_DATA_FILE_NAME = 'dataset_with_lyrics.csv'

_LANGUAGE_CODE = 'en'

_ACCESS_TOKEN = 'T1veoJBfvive92a4D9m2gjkmhOwNXP91-xbwpkgXFl3ws2nqUlBOoTzTZ3vNhEPo'
genius = lyricsgenius.Genius(_ACCESS_TOKEN)
genius.verbose = True


def get_lyrics(df: pd.DataFrame) -> pd.DataFrame:
    print("Getting lyrics")

    df['lyrics'] = None
    for i, row in df.iterrows():
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S}\t{i + 1}/{len(df)}")
        if pd.isna(row['lyrics']):
            lyrics = get_lyric_for_row(row)
            df.at[i, 'lyrics'] = lyrics

    out_df = df.dropna(subset=['lyrics'])
    return out_df


def filter_language(df: pd.DataFrame, language_code: str) -> pd.DataFrame:
    df['language'] = np.nan
    df['language'] = df['lyrics'].apply(lambda lyrics: get_language(lyrics))
    out_df = df[df['language'].isin([language_code]) & ~df['lyrics'].isnull()]
    return out_df


def get_genres(df: pd.DataFrame) -> None:
    print("Getting genres")
    for line_number, (index, row) in enumerate(df.iterrows()):
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S}\t{line_number + 1}/{len(df)}")
        if pd.isna(row['genre']) or row['genre'] == 'null':
            genre = get_genres_for_row(row)
            df.at[index, 'genre'] = genre


def get_lyric_for_row(row: pd.Series) -> str:
    title = row['title']
    artist = row['artist']
    try:
        song = genius.search_song(title, artist)
    except Exception as e:
        print(e)
        song = None
    if song is not None and song.artist != 'Spotify':
        song_lyrics = re.sub(r'\n', ' ', song.lyrics)
    else:
        song_lyrics = None
    return song_lyrics


def get_language(lyrics: str) -> str:
    if lyrics is None:
        lang = None
    else:
        lang = detect(lyrics)
    return lang


def get_genres_for_row(row: pd.Series) -> Optional[str]:
    title = row['title']
    artist = row['artist']
    http_link = get_http_link(artist, title)
    genre = None
    try:
        response = requests.get(http_link)
        response.raise_for_status()
        json_response = response.json()
        genre = json_response['track'][0]['strGenre']
    except HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    return genre


def get_http_link(artist_query, title_query):
    artist_query = artist_query.replace(' ', '_').lower()
    title_query = title_query.replace(' ', '_').lower()
    http_link = 'http://theaudiodb.com/api/v1/json/1/searchtrack.php?s=' + artist_query + '&t=' + title_query
    return http_link


if __name__ == '__main__':
    song_df = pd.read_csv(os.path.join(_DATASET_PATH, _DATA_FILE), index_col=0)
    song_df = get_lyrics(song_df)
    song_df = filter_language(song_df, _LANGUAGE_CODE)
    get_genres(song_df)

    song_df.to_csv(os.path.join(_DATASET_PATH, _NEW_DATA_FILE_NAME))
