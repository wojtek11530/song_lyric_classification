import os
import pickle as pkl
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split

from models.conv_net.conv_net_model import ConvNetClassifier
from models.lstm.lstm_model import LSTMClassifier
from models.mlp.mlp_model import MLPClassifier
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
    result_dict = {'MLP': defaultdict(list), 'LSTM': defaultdict(list), 'CNN': defaultdict(list)}

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
        print(f"{datetime.now().strftime('%m-%d-%Y_%H.%M')} Ratio {ratio} - simulation {i + 1}/{simulation_number}")
        print("CNN")
        df_to_train_copy = df_to_train.copy()
        cnn_model = _train_conv_net(df_to_train_copy, ratio)
        cnn_class_report = get_classification_report_for_evaluation(cnn_model)
        result_dict['CNN'][ratio].append(cnn_class_report)
        print("MLP")
        df_to_train_copy = df_to_train.copy()
        mlp_model = _train_mlp(df_to_train_copy, ratio)
        mlp_class_report = get_classification_report_for_evaluation(mlp_model)
        result_dict['MLP'][ratio].append(mlp_class_report)
        print("LSTM")
        df_to_train_copy = df_to_train.copy()
        lstm_model = _train_lstm(df_to_train_copy, ratio)
        lstm_class_report = get_classification_report_for_evaluation(lstm_model)
        result_dict['LSTM'][ratio].append(lstm_class_report)


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
    trainer = _get_trainer(logger, max_epochs=120)
    model = ConvNetClassifier(**hp)
    trainer.fit(model)
    return model


def _train_mlp(df: pd.DataFrame, ratio: float) -> MLPClassifier:
    hp = {
        'input_size': 200,
        'output_size': 4,
        'dropout': 0.5,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'removing_stop_words': True,
        'lemmatization': False,
        'smote': False,
        'train_df': df
    }
    name = f'MLP_ratio={ratio}'
    logger = _get_tensor_board_logger(name)
    trainer = _get_trainer(logger, max_epochs=120)
    model = MLPClassifier(**hp)
    trainer.fit(model)
    return model


def _train_lstm(df: pd.DataFrame, ratio: float) -> LSTMClassifier:
    hp = {
        'input_dim': 200,
        'hidden_dim': 200,
        'output_dim': 4,
        'layer_dim': 1,
        'bidirectional': False,
        'dropout': 0.0,
        'batch_size': 128,
        'learning_rate': 9e-5,
        'weight_decay': 1e-4,
        'max_num_words': 200,
        'removing_stop_words': True,
        'lemmatization': False,
        'train_df': df
    }
    name = f'MLP_ratio={ratio}'
    logger = _get_tensor_board_logger(name)
    trainer = _get_trainer(logger, max_epochs=80)
    model = LSTMClassifier(**hp)
    trainer.fit(model)
    return model


def _get_tensor_board_logger(name) -> TensorBoardLogger:
    logger = TensorBoardLogger(
        name=name,
        save_dir=os.path.join(os.getcwd(), 'lightning_logs', 'Analysis')
    )
    return logger


def _get_trainer(logger: TensorBoardLogger, max_epochs: int) -> pl.Trainer:
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=max_epochs,
        early_stop_callback=EarlyStopping(monitor='val_loss', mode='min', patience=6, verbose=True),
        gpus=1
    )
    return trainer


def plot_results():
    file_name = 'size_of_training_dataset_analysis_results_11-05-2020_11.53.pkl'
    with open(file_name, 'rb') as handle:
        training_data_dictionary = pkl.load(handle)
        _plot_average_results(training_data_dictionary)


def _plot_average_results(training_data_dictionary: Dict[str, Dict[float, List[Dict]]]):
    for model_name, ratios_dict in training_data_dictionary.items():
        ratios = []
        avg_accuracy = []
        avg_f1_score = []
        for ratio, classification_reports_list in ratios_dict.items():
            accuracies = [report['accuracy'] for report in classification_reports_list]
            f1_scores = [report['macro avg']['f1-score'] for report in classification_reports_list]

            ratios.append(ratio)
            avg_accuracy.append(np.mean(accuracies))
            avg_f1_score.append(np.mean(f1_scores))

        plt.plot(ratios, avg_accuracy, 'X--', ms=9, label=model_name)

    plt.xlim((0, 1.1))
    locs, labels = plt.xticks()
    plt.xticks(locs[:-1], ['{:.0%}'.format(x) for x in locs[:-1]])
    plt.grid(axis='y')
    plt.legend(loc=4)
    plt.ylabel('Średnia dokładność modelu (avg. accuracy)')
    plt.xlabel('Użycie zbioru treningowego')
    plt.tight_layout()
    plt.show()


def merge_results():
    file_name_1 = 'size_of_training_dataset_analysis_results_11-04-2020_17.18.pkl'
    file_name_2 = 'size_of_training_dataset_analysis_results_11-05-2020_11.00.pkl'
    with open(file_name_1, 'rb') as handle_1, open(file_name_2, 'rb') as handle_2:
        training_data_dictionary_1 = pkl.load(handle_1)
        training_data_dictionary_2 = pkl.load(handle_2)
        data_dictionary = dict(training_data_dictionary_1)
        data_dictionary.update(training_data_dictionary_2)

        file_name = f'size_of_training_dataset_analysis_results_{datetime.now().strftime("%m-%d-%Y_%H.%M")}.pkl'
        with open(file_name, 'wb') as handle:
            pkl.dump(data_dictionary, handle, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # run_analysis()
    plot_results()
    # merge_results()
