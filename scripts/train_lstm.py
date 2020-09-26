import os
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping

from models.lstm.lstm_model import LSTMClassifier

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def run_train_lstm():
    my_trainer = pl.Trainer(
        max_epochs=100,
        early_stop_callback=EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=True),
        gpus=1
    )
    model = LSTMClassifier(
        input_dim=200,
        output_dim=4,
        bidirectional=False,
        dropout=0.5,
        batch_size=32,
        layer_dim=1,
        learning_rate=1e-4,
        weight_decay=5e-3
    )
    my_trainer.fit(model)
    model_name = 'saved_lstm_model_' + datetime.now().strftime('%m-%d-%Y_%H.%M.%S') + '.pt'
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'lstm', model_name)
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    run_train_lstm()
