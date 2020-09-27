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
        'input_dim': 100,
        'output_dim': 4,
        'bidirectional': False,
        'dropout': 0.5,
        'batch_size': 32,
        'layer_dim': 1,
        'learning_rate': 1e-4,
        'weight_decay': 5e-3
    }
    name = get_tensorboard_log_name(hp)
    logger = TensorBoardLogger(
        name=name,
        save_dir=os.path.join(os.getcwd(), 'lightning_logs')
    )

    my_trainer = pl.Trainer(
        logger=logger,
        max_epochs=100,
        early_stop_callback=EarlyStopping(monitor='val_loss', mode='min', patience=6, verbose=True),
        gpus=1
    )
    model = LSTMClassifier(**hp)
    my_trainer.fit(model)
    model_name = name + '_' + datetime.now().strftime('%m-%d-%Y_%H.%M.%S') + '.pt'
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'lstm', model_name)
    torch.save(model.state_dict(), model_path)


def get_tensorboard_log_name(hp: Dict[str, Union[float, bool]]) -> str:
    name = 'LSTM_input_' + str(hp['input_dim']) + '_drop_' + str(hp['dropout']) + '_lay_num_' + \
           str(hp['layer_dim']) + '_lr_' + str(hp['learning_rate']) + '_wd_' + str(hp['weight_decay'])
    if hp['bidirectional']:
        name = 'Bi' + name
    return name


if __name__ == '__main__':
    run_train_lstm()
