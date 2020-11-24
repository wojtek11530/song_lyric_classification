import os
from datetime import datetime
from typing import Dict, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from models.lstm.lstm_model import LSTMClassifier

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def run_train_lstm():
    hp = {
        'input_dim': 200,
        'hidden_dim': 200,
        'output_dim': 4,
        'layer_dim': 1,
        'bidirectional': False,
        'dropout': 0.5,
        'batch_size': 128,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'max_num_words': 200,
        'removing_stop_words': True,
        'lemmatization': False
    }
    name = get_tensorboard_log_name(hp)
    logger = TensorBoardLogger(
        name=name,
        save_dir=os.path.join(os.getcwd(), '../lightning_logs', 'LSTM')
    )

    my_trainer = pl.Trainer(
        logger=logger,
        max_epochs=40,
        early_stop_callback=EarlyStopping(monitor='val_loss', mode='min', patience=6, verbose=True),
        gpus=1
    )

    model = LSTMClassifier(**hp)
    my_trainer.fit(model)
    model_name = name + '_' + datetime.now().strftime('%m-%d-%Y_%H.%M.%S') + '.pt'
    project_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(project_directory, 'models', 'lstm', 'saved_models', model_name)
    torch.save(model.state_dict(), model_path)


def get_tensorboard_log_name(hp: Dict[str, Union[float, bool]]) -> str:
    name = 'LSTM_input_' + str(hp['input_dim']) + '_hidden_' + str(hp['hidden_dim']) + '_drop_' \
           + str(hp['dropout']) + '_lay_num_' + str(hp['layer_dim']) + '_lr_' + str(hp['learning_rate']) \
           + '_wd_' + str(hp['weight_decay']) + '_max_words_' + str(hp['max_num_words']) + '_rem_sw_' \
           + str(hp['removing_stop_words']) + '_lemm_' + str(hp['lemmatization'])
    if hp['bidirectional']:
        name = 'Bi' + name
    return name


if __name__ == '__main__':
    run_train_lstm()
