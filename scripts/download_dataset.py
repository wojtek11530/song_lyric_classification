import os
import shutil
import zipfile

import gdown

_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATASET_PATH = os.path.join(_PROJECT_PATH, 'datasets')
_SCRIPTS_DIRECTORY = os.path.join(_PROJECT_PATH, 'scripts')


def download_pmemo_dataset():
    pmemo_url = 'https://drive.google.com/uc?id=1UzC3NCDj30j9Ba7i5lkMzWO5gFqSr0OJ'
    pmemo_dataset_name = 'PMEmo2019'
    pmmemo_output = os.path.join(_SCRIPTS_DIRECTORY, pmemo_dataset_name + '.zip')
    pmemo_directory = os.path.join(_DATASET_PATH, pmemo_dataset_name)
    if not os.path.isdir(pmemo_directory):
        os.mkdir(pmemo_directory)

    # downloading from Google Drive, file size: 648 MB
    gdown.download(pmemo_url, pmmemo_output, quiet=False)
    with zipfile.ZipFile(pmmemo_output, 'r') as zip_ref:
        zip_ref.extract(member=pmemo_dataset_name + '/metadata.csv')
        zip_ref.extract(member=pmemo_dataset_name + '/annotations/static_annotations.csv')
        zip_ref.extract(member=pmemo_dataset_name + '/annotations/static_annotations_std.csv')

    shutil.copy(os.path.join(_SCRIPTS_DIRECTORY, pmemo_dataset_name, 'metadata.csv'),
                pmemo_directory)
    shutil.copy(os.path.join(_SCRIPTS_DIRECTORY, pmemo_dataset_name, 'annotations', 'static_annotations.csv'),
                pmemo_directory)
    shutil.copy(os.path.join(_SCRIPTS_DIRECTORY, pmemo_dataset_name, 'annotations', 'static_annotations_std.csv'),
                pmemo_directory)

    os.remove(pmmemo_output)
    shutil.rmtree(os.path.join(_SCRIPTS_DIRECTORY, pmemo_dataset_name), ignore_errors=True)


if __name__ == '__main__':
    download_pmemo_dataset()
