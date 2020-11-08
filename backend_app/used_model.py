import os

import torch

from models.base import BaseModel
from models.conv_net.conv_net_model import ConvNetClassifier

_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_CONV_MODEL_PATH = os.path.join(
    _PROJECT_PATH, 'models', 'conv_net', 'saved_models',
    'ConvNet_embed_200_filters_num_128_kern_[5, 10, 15]_drop_0.4_lr_0.0001_wd_0.0003_max_words_256_rem_sw_True_lemm_False.pt'
)


def get_model() -> BaseModel:
    cnn_model = _get_cnn_model()
    return cnn_model


def _get_cnn_model() -> ConvNetClassifier:
    conv_model = ConvNetClassifier(
        embedding_dim=200,
        output_dim=4,
        dropout=0.5,
        batch_size=128,
        learning_rate=1e-4,
        weight_decay=65e-4,
        filters_number=128,
        kernels_sizes=[5, 10, 15],
        max_num_words=256,
        removing_stop_words=True,
        lemmatization=False
    )
    conv_model.load_state_dict(torch.load(_CONV_MODEL_PATH, map_location=_DEVICE))
    conv_model.eval()
    return conv_model
