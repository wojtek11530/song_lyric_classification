import os
from datetime import datetime
from typing import Dict, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from models.conv_net.conv_net_model import ConvNetClassifier

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def run_train_conv_net():
    hp = {
        'embedding_dim': 200,
        'output_dim': 4,
        'dropout': 0.5,
        'batch_size': 32,
        'learning_rate': 5e-3,
        'weight_decay': 5e-3,
        'max_num_words': 200,
        'removing_stop_words': True
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
    model = ConvNetClassifier(**hp)
    my_trainer.fit(model)
    model_name = name + '_' + datetime.now().strftime('%m-%d-%Y_%H.%M.%S') + '.pt'
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'conv_net', model_name)
    torch.save(model.state_dict(), model_path)


def get_tensorboard_log_name(hp: Dict[str, Union[float, bool]]) -> str:
    name = 'ConvNet_embed_' + str(hp['embedding_dim']) + '_drop_' + str(hp['dropout']) + '_lr_' + str(hp['learning_rate']) \
           + '_wd_' + str(hp['weight_decay']) + '_max_words_' + str(hp['max_num_words']) + \
           '_rem_sw_' + str(hp['removing_stop_words'])
    return name


if __name__ == '__main__':
    run_train_conv_net()
