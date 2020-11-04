import os
import pickle as pkl
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool
from typing import Dict

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split

from models.conv_net.conv_net_model import ConvNetClassifier
from models.word_embedding.word_embedder import WordEmbedder
from scripts.evaluation.evaluate_model import get_classification_report_for_evaluation

_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATASET_PATH = os.path.join(_PROJECT_PATH, 'datasets')
_DATA_FILE = 'train_dataset.csv'

_RANDOM_SEED = 42

train_df = pd.read_csv(os.path.join(_DATASET_PATH, _DATA_FILE), index_col=0)

simulation_number = 5
dataset_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def run_analysis():
    WordEmbedder()
    result_dict = {'CNN': defaultdict(list)}

    for ratio in dataset_ratios:
        perform_trainings_for_ratio(ratio, result_dict)

    file_name = f'size_of_training_dataset_analysis_results_{datetime.now().strftime("%m-%d-%Y_%H.%M")}.pkl'
    with open(file_name, 'wb') as handle:
        pkl.dump(result_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)


def perform_trainings_for_ratio(ratio: float, result_dict: Dict):
    if ratio >= 1:
        df_to_train = train_df
    else:
        df_to_train, _ = train_test_split(
            train_df,
            test_size=1 - ratio,
            random_state=_RANDOM_SEED,
            stratify=train_df['emotion_4Q']
        )
    print(df_to_train['emotion_4Q'].value_counts())
    for i in range(simulation_number):
        print(f'Ratio {ratio} - simulation {i + 1}/{simulation_number}')
        print('CNN')
        df_to_train_copy = df_to_train.copy()
        cnn_model = _train_conv_net(df_to_train_copy, ratio)
        cnn_class_report = get_classification_report_for_evaluation(cnn_model)
        result_dict['CNN'][ratio].append(cnn_class_report)


def _train_conv_net(df: pd.DataFrame, ratio: float) -> ConvNetClassifier:
    hp = {
        'embedding_dim': 200,
        'output_dim': 4,
        'dropout': 0.4,
        'batch_size': 128,
        'learning_rate': 1e-4,
        'weight_decay': 3e-4,
        'filters_number': 128,
        'kernels_sizes': [5, 10, 15],
        'max_num_words': 256,
        'removing_stop_words': True,
        'lemmatization': False,
        'smote': False,
        'train_df': df
    }
    name = f'ConvNet_ratio={ratio}'
    logger = _get_tensor_board_logger(name)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=80,
        early_stop_callback=EarlyStopping(monitor='val_loss', mode='min', patience=6, verbose=True),
        gpus=1
    )
    model = ConvNetClassifier(**hp)
    trainer.fit(model)
    return model


def _get_tensor_board_logger(name) -> TensorBoardLogger:
    logger = TensorBoardLogger(
        name=name,
        save_dir=os.path.join(os.getcwd(), 'lightning_logs', 'Analysis')
    )
    return logger


def load_results():
    file_name = 'size_of_training_dataset_analysis_results_11-04-2020_12.20.pkl'
    with open(file_name, 'rb') as handle:
        training_data_dictionary = pkl.load(handle)
        pass


if __name__ == '__main__':
    # run_analysis()
    load_results()
