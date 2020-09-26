import os
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping

from models.mlp.mlp_model import MLPClassifier


def run_train_mlp():
    my_trainer = pl.Trainer(
        max_epochs=100,
        early_stop_callback=EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=True),
        gpus=1
    )
    model = MLPClassifier(input_size=200, output_size=4, dropout=0.5, weight_decay=1e-5, batch_size=64)
    my_trainer.fit(model)
    model_name = 'saved_mlp_model_' + datetime.now().strftime('%m-%d-%Y_%H.%M.%S') + '.pt'
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'mlp', model_name)
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    run_train_mlp()
