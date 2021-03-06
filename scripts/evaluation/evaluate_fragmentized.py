import os

import torch

from models.conv_net.fragmentized_conv_net_model import FragmentizedConvNetClassifier
from models.lstm.fragmentized_lstm_model import FragmentizedLSTMClassifier
from models.mlp.fragmentized_mlp_model import FragmentizedMLPClassifier
from scripts.evaluation.evaluate_model import evaluate_model

_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_MLP_MODEL_PATH = os.path.join(
    _PROJECT_PATH, 'models', 'mlp', 'saved_models',
    'FragMLP_input_200_drop_0.5_lr_0.001_wd_1e-05_rem_sw_True_lemm_False.pt'
)


def evaluate_fragmentized_mlp():
    print('FragMLP model')
    mlp_model = FragmentizedMLPClassifier(input_size=200, output_size=4, dropout=0.5, weight_decay=5e-3,
                                          batch_size=128, removing_stop_words=True, lemmatization=False)
    mlp_model.load_state_dict(torch.load(_MLP_MODEL_PATH, map_location=device))
    evaluate_model(mlp_model)


_CONV_MODEL_PATH = os.path.join(
    _PROJECT_PATH, 'models', 'conv_net', 'saved_models',
    'FragConvNet_embed_200_filters_num_64_kern_[5, 10, 15]_drop_0.4_lr_0.0003_wd_0.0003_max_words_64'
    '_rem_sw_True_lemm_False.pt'
)


def evaluate_fragmentized_conv_net():
    print('FragConvNet model')
    conv_model = FragmentizedConvNetClassifier(
        embedding_dim=200,
        output_dim=4,
        dropout=0.5,
        batch_size=128,
        learning_rate=1e-4,
        weight_decay=65e-4,
        filters_number=64,
        kernels_sizes=[5, 10, 15],
        max_num_words=64,
        removing_stop_words=True,
        lemmatization=False
    )
    conv_model.load_state_dict(torch.load(_CONV_MODEL_PATH, map_location=device))
    evaluate_model(conv_model)


_LSTM_MODEL_PATH = os.path.join(
    _PROJECT_PATH, 'models', 'lstm', 'saved_models',
    'FragLSTM_input_200_hidden_200_drop_0.0_lay_num_1_lr_9e-05_wd_0.0001_max_words_64_rem_sw_True_lemm_False.pt'
)


def evaluate_fragmentized_lstm():
    print('FragLSTM model')
    lstm_model = FragmentizedLSTMClassifier(
        input_dim=200,
        hidden_dim=200,
        output_dim=4,
        layer_dim=1,
        bidirectional=False,
        dropout=0.0,
        batch_size=64,
        learning_rate=5e-3,
        weight_decay=5e-3,
        max_num_words=64,
        removing_stop_words=True,
        lemmatization=False
    )
    lstm_model.load_state_dict(torch.load(_LSTM_MODEL_PATH, map_location=device))
    evaluate_model(lstm_model)


if __name__ == '__main__':
    evaluate_fragmentized_mlp()
    evaluate_fragmentized_lstm()
    evaluate_fragmentized_conv_net()
