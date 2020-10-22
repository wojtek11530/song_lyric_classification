import os
from datetime import datetime
from typing import Dict, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from models.mlp.fragmentized_mlp_model import FragmentizedMLPClassifier


def run_train_fragmentized_mlp():
    hp = {
        'input_size': 200,
        'output_size': 4,
        'dropout': 0.5,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'removing_stop_words': True,
        'lemmatization': False
    }
    name = get_tensorboard_log_name(hp)
    logger = TensorBoardLogger(
        name=name,
        save_dir=os.path.join(os.getcwd(), '../lightning_logs', 'MLP')
    )

    my_trainer = pl.Trainer(
        logger=logger,
        max_epochs=50,
        early_stop_callback=EarlyStopping(monitor='val_loss', mode='min', patience=6, verbose=True),
        gpus=1
    )
    model = FragmentizedMLPClassifier(**hp)
    my_trainer.fit(model)
    model_name = name + '_' + datetime.now().strftime('%m-%d-%Y_%H.%M.%S') + '.pt'
    project_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(project_directory, 'models', 'mlp', 'saved_models', model_name)
    torch.save(model.state_dict(), model_path)


def get_tensorboard_log_name(hp: Dict[str, Union[float, bool]]) -> str:
    name = 'FragMLP_input_' + str(hp['input_size']) + '_drop_' + str(hp['dropout']) + '_lr_' + \
           str(hp['learning_rate']) + '_wd_' + str(hp['weight_decay']) + '_rem_sw_' \
           + str(hp['removing_stop_words']) + '_lemm_' + str(hp['lemmatization'])
    return name


if __name__ == '__main__':
    run_train_fragmentized_mlp()
