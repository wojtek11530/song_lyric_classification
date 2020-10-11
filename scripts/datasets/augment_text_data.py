import os
import re

import nlpaug.augmenter.word as naw
import pandas as pd
from tqdm import tqdm

_PROJECT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATASET_PATH = os.path.join(_PROJECT_DIRECTORY, 'datasets')
_TRAIN_DATASET_FILEPATH = os.path.join(_DATASET_PATH, 'train_dataset.csv')

train_df = pd.read_csv(_TRAIN_DATASET_FILEPATH, index_col=0)

aug = naw.SynonymAug(aug_src='wordnet')

train_aug = train_df.copy()

tqdm.pandas()
train_aug['lyrics'] = train_aug['lyrics'].progress_apply(lambda x: re.sub(r'\s+\'\s+', '\'', str(aug.augment(x))))

train_df = train_df.append(train_aug, ignore_index=True)

train_df.to_csv(os.path.join(_DATASET_PATH, 'train_aug_dataset.csv'))
