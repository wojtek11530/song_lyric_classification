import os
from datetime import datetime
from typing import Dict, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from models.gru_cnn.gru_cnn_model import GRUCNNClassifier

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def run_train_gru():
    hp = {
        'input_dim': 200,
        'output_dim': 4,
        'gru_hidden_dim': 200,
        'gru_layer_dim': 1,
        'filters_number': 128,
        'kernels_sizes': [5, 10, 15],
        'dropout': 0.3,
        'batch_size': 128,
        'learning_rate': 2e-4,
        'weight_decay': 5e-4,
        'max_num_words': 200,
        'removing_stop_words': True,
        'lemmatization': False
    }
    name = get_tensorboard_log_name(hp)
    logger = TensorBoardLogger(
        name=name,
        save_dir=os.path.join(os.getcwd(), '../lightning_logs', 'GRUCNN')
    )

    my_trainer = pl.Trainer(
        logger=logger,
        max_epochs=60,
        early_stop_callback=EarlyStopping(monitor='val_loss', mode='min', patience=6, verbose=True),
        gpus=1
    )

    model = GRUCNNClassifier(**hp)
    my_trainer.fit(model)
    model_name = name + '_' + datetime.now().strftime('%m-%d-%Y_%H.%M.%S') + '.pt'
    project_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(project_directory, 'models', 'gru_cnn', 'saved_models', model_name)
    torch.save(model.state_dict(), model_path)


def get_tensorboard_log_name(hp: Dict[str, Union[float, bool]]) -> str:
    name = 'GRUCNN_input_' + str(hp['input_dim']) + '_drop_' + str(hp['dropout']) + '_hidden_' + \
           str(hp['gru_hidden_dim']) + '_lay_num_' + str(hp['gru_layer_dim']) + '_filters_num_' + \
           str(hp['filters_number']) + '_kern_' + str(hp['kernels_sizes']) + '_lr_' + str(hp['learning_rate']) \
           + '_wd_' + str(hp['weight_decay']) + '_max_words_' + str(hp['max_num_words']) + '_rem_sw_' \
           + str(hp['removing_stop_words']) + '_lemm_' + str(hp['lemmatization'])
    return name


if __name__ == '__main__':
    run_train_gru()
