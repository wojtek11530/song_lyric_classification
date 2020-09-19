import os
import shutil
import zipfile

import gdown
import requests

_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATASET_PATH = os.path.join(_PROJECT_PATH, 'datasets')
_SCRIPTS_DIRECTORY = os.path.join(_PROJECT_PATH, 'scripts')

if not os.path.isdir(_DATASET_PATH):
    os.mkdir(_DATASET_PATH)


def download_pmemo_dataset():
    pmemo_url = 'https://drive.google.com/uc?id=1UzC3NCDj30j9Ba7i5lkMzWO5gFqSr0OJ'
    pmemo_dataset_name = 'PMEmo2019'
    pmemo_output = os.path.join(_SCRIPTS_DIRECTORY, pmemo_dataset_name + '.zip')
    pmemo_directory = os.path.join(_DATASET_PATH, pmemo_dataset_name)
    if not os.path.isdir(pmemo_directory):
        os.mkdir(pmemo_directory)

    # downloading from Google Drive, file size: 648 MB
    gdown.download(pmemo_url, pmemo_output, quiet=False)
    with zipfile.ZipFile(pmemo_output, 'r') as zip_ref:
        zip_ref.extract(member=pmemo_dataset_name + '/metadata.csv')
        zip_ref.extract(member=pmemo_dataset_name + '/annotations/static_annotations.csv')
        zip_ref.extract(member=pmemo_dataset_name + '/annotations/static_annotations_std.csv')

    shutil.copy(os.path.join(_SCRIPTS_DIRECTORY, pmemo_dataset_name, 'metadata.csv'),
                pmemo_directory)
    shutil.copy(os.path.join(_SCRIPTS_DIRECTORY, pmemo_dataset_name, 'annotations', 'static_annotations.csv'),
                pmemo_directory)
    shutil.copy(os.path.join(_SCRIPTS_DIRECTORY, pmemo_dataset_name, 'annotations', 'static_annotations_std.csv'),
                pmemo_directory)

    os.remove(pmemo_output)
    shutil.rmtree(os.path.join(_SCRIPTS_DIRECTORY, pmemo_dataset_name), ignore_errors=True)


def download_emomusic_dataset():
    emomusic_url = 'http://cvml.unige.ch/databases/emoMusic/annotations.tar.gz'
    emomusic_dataset_name = 'emoMusic'
    emomusic_output = os.path.join(_SCRIPTS_DIRECTORY, emomusic_dataset_name + '.tar.gz')
    emomusic_directory = os.path.join(_DATASET_PATH, emomusic_dataset_name)
    if not os.path.isdir(emomusic_directory):
        os.mkdir(emomusic_directory)

    r = requests.get(emomusic_url, allow_redirects=True)
    open(emomusic_output, 'wb').write(r.content)

    shutil.unpack_archive(emomusic_output, os.path.join(_SCRIPTS_DIRECTORY, emomusic_dataset_name))
    shutil.copy(os.path.join(_SCRIPTS_DIRECTORY, emomusic_dataset_name, 'songs_info.csv'),
                emomusic_directory)
    shutil.copy(os.path.join(_SCRIPTS_DIRECTORY, emomusic_dataset_name, 'static_annotations.csv'),
                emomusic_directory)

    os.remove(emomusic_output)
    shutil.rmtree(os.path.join(_SCRIPTS_DIRECTORY, emomusic_dataset_name), ignore_errors=True)


def download_MoodyLyrics4Q_dataset():
    MoodyLyrics4Q_url = 'http://softeng.polito.it/erion/MoodyLyrics4Q.zip'
    MoodyLyrics4Q_dataset_name = 'MoodyLyrics4Q'
    MoodyLyrics4Q_output = os.path.join(_SCRIPTS_DIRECTORY, MoodyLyrics4Q_dataset_name + '.zip')
    MoodyLyrics4Q_directory = os.path.join(_DATASET_PATH, MoodyLyrics4Q_dataset_name)
    if not os.path.isdir(MoodyLyrics4Q_directory):
        os.mkdir(MoodyLyrics4Q_directory)

    r = requests.get(MoodyLyrics4Q_url, allow_redirects=True)
    open(MoodyLyrics4Q_output, 'wb').write(r.content)

    shutil.unpack_archive(MoodyLyrics4Q_output, os.path.join(_SCRIPTS_DIRECTORY))
    shutil.copy(os.path.join(_SCRIPTS_DIRECTORY, MoodyLyrics4Q_dataset_name, 'MoodyLyrics4Q.csv'),
                MoodyLyrics4Q_directory)

    os.remove(MoodyLyrics4Q_output)
    shutil.rmtree(os.path.join(_SCRIPTS_DIRECTORY, MoodyLyrics4Q_dataset_name), ignore_errors=True)


if __name__ == '__main__':
    download_pmemo_dataset()
    download_emomusic_dataset()
    download_MoodyLyrics4Q_dataset()
