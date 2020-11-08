import os

import torch

from models.base import BaseModel
from models.conv_net.conv_net_model import ConvNetClassifier
from models.gru.gru_model import GRUClassifier
from models.label_encoder import label_encoder
from models.lstm.lstm_model import LSTMClassifier
from models.mlp.mlp_model import MLPClassifier

_CLASS_NAMES = label_encoder.classes_

_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_LYRICS = "Here comes the sun, do, dun, do, do Here comes the sun, and I say It's all right Little darling, " \
          "it's been a long cold lonely winter Little darling, it feels like years since it's been here Here comes " \
          "the sun, do, dun, do, do Here comes the sun, and I say It's all right Little darling, the smile's " \
          "returning to their faces Little darling, it seems like years since it's been here Here comes the sun, do, " \
          "dun, do, do Here comes the sun, and I say It's all right Sun, sun, sun, here it comes Sun, sun, sun, " \
          "here it comes Sun, sun, sun, here it comes Sun, sun, sun, here it comes Sun, sun, sun, here it comes " \
          "Little darling, I feel that ice is slowly melting Little darling, it seems like years since it's been " \
          "clear Here comes the sun, do, dun, do, do Here comes the sun, and I say It's all right Here comes the sun, " \
          "do, dun, do, do Here comes the sun It's all right It's all right"

_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_MLP_MODEL_PATH = os.path.join(
    _PROJECT_PATH, 'models', 'mlp', 'saved_models',
    'MLP_input_200_drop_0.5_lr_0.001_wd_1e-05_rem_sw_True_lemm_False.pt'
)

_LSTM_MODEL_PATH = os.path.join(
    _PROJECT_PATH, 'models', 'lstm', 'saved_models',
    'LSTM_input_200_hidden_200_drop_0.0_lay_num_1_lr_9e-05_wd_0.0001_max_words_200_rem_sw_True_lemm_False.pt'
)

_GRU_MODEL_PATH = os.path.join(
    _PROJECT_PATH, 'models', 'gru', 'saved_models',
    'GRU_input_200_hidden_200_drop_0.0_lay_num_1_lr_9e-05_wd_1e-05_max_words_200_rem_sw_True_lemm_False.pt'
)

_CONV_MODEL_PATH = os.path.join(
    _PROJECT_PATH, 'models', 'conv_net', 'saved_models',
    'ConvNet_embed_200_filters_num_128_kern_[5, 10, 15]_drop_0.4_lr_0.0001_wd_0.0003_max_words_256_rem_sw_True_lemm_False.pt'
)


def perform_prediction():
    mlp_model = get_mlp_model()
    lstm_model = get_lstm_model()
    gru_model = get_gru_model()
    cnn_model = get_cnn_model()

    mlp_result = predict_emotion(mlp_model, _LYRICS)
    lstm_result = predict_emotion(lstm_model, _LYRICS)
    gru_result = predict_emotion(gru_model, _LYRICS)
    cnn_result = predict_emotion(cnn_model, _LYRICS)

    print(f'Lyrics: {_LYRICS}')
    print(f'Predicted emotion:)'
          f'\n - mlp: {mlp_result}'
          f'\n - lstm: {lstm_result}'
          f'\n - gru: {gru_result}'
          f'\n - cnn: {cnn_result}')


def predict_emotion(model: BaseModel, lyrics: str) -> str:
    encoded_label, probs = model.predict(lyrics)
    label = label_encoder.inverse_transform(encoded_label)
    return label[0]


def get_mlp_model() -> MLPClassifier:
    mlp_model = MLPClassifier(input_size=200, output_size=4, dropout=0.5, weight_decay=5e-3,
                              batch_size=128, removing_stop_words=True, lemmatization=False)
    mlp_model.load_state_dict(torch.load(_MLP_MODEL_PATH, map_location=_DEVICE))
    mlp_model.eval()
    return mlp_model


def get_lstm_model() -> LSTMClassifier:
    lstm_model = LSTMClassifier(
        input_dim=200,
        hidden_dim=200,
        output_dim=4,
        layer_dim=1,
        bidirectional=False,
        dropout=0.0,
        batch_size=64,
        learning_rate=5e-3,
        weight_decay=5e-3,
        max_num_words=200,
        removing_stop_words=True,
        lemmatization=False
    )
    lstm_model.load_state_dict(torch.load(_LSTM_MODEL_PATH, map_location=_DEVICE))
    lstm_model.eval()
    return lstm_model


def get_gru_model() -> GRUClassifier:
    gru_model = GRUClassifier(
        input_dim=200,
        hidden_dim=200,
        output_dim=4,
        layer_dim=1,
        dropout=0.0,
        batch_size=16,
        learning_rate=5e-3,
        weight_decay=5e-3,
        max_num_words=200,
        removing_stop_words=True,
        lemmatization=False
    )
    gru_model.load_state_dict(torch.load(_GRU_MODEL_PATH, map_location=_DEVICE))
    gru_model.eval()
    return gru_model


def get_cnn_model() -> ConvNetClassifier:
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


if __name__ == '__main__':
    perform_prediction()
