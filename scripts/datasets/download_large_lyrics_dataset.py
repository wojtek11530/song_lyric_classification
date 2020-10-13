import os

import kaggle.api

_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATASET_PATH = os.path.join(_PROJECT_PATH, 'datasets', 'lyrics-data', )
_KAGGLE_DATASET_NAME = 'neisse/scrapped-lyrics-from-6-genres'

# Ensure you have json file kaggle.json with your kaggle account token in your HOME directory
kaggle.api.authenticate()

if not os.path.exists(_DATASET_PATH):
    os.makedirs(_DATASET_PATH)

kaggle.api.dataset_download_files(_KAGGLE_DATASET_NAME, path=_DATASET_PATH, unzip=True)
