import os

import torch

from models.base import BaseModel
from models.conv_net.fragmentized_conv_net_model import FragmentizedConvNetClassifier
from models.label_encoder import label_encoder
from models.lstm.fragmentized_lstm_model import FragmentizedLSTMClassifier
from models.mlp.fragmentized_mlp_model import FragmentizedMLPClassifier

_CLASS_NAMES = label_encoder.classes_

_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_LYRICS = "Here comes the sun, do, dun, do, do Here comes the sun, and I say It's all right Little darling, " \
          "it's been a long cold lonely winter Little darling, it feels like years since it's been here Here comes " \
          "the sun, do, dun, do, do Here comes the sun, and I say It's all right Little darling, the smile's " \
          "returning to their faces Little darling, it seems like years since it's been here Here comes the sun, do, " \
          "dun, do, do Here comes the sun, and I say It's all right Sun, sun, sun, here it comes Sun, sun, sun, " \
          "here it comes Sun, sun, sun, here it comes Sun, sun, sun, here it comes Sun, sun, sun, here it comes " \
          "Little darling, I feel that ice is slowly melting Little darling, it seems like years since it's been " \
          "clear Here comes the sun, do, dun, do, do Here comes the sun, and I say It's all right Here comes the " \
          "sun, do, dun, do, do Here comes the sun It's all right It's all right"

_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_MLP_MODEL_PATH = os.path.join(
    _PROJECT_PATH, 'models', 'mlp', 'saved_models',
    'FragMLP_input_200_drop_0.5_lr_0.001_wd_1e-05_rem_sw_True_lemm_False.pt'
)

_LSTM_MODEL_PATH = os.path.join(
    _PROJECT_PATH, 'models', 'lstm', 'saved_models',
    'FragLSTM_input_200_hidden_200_drop_0.0_lay_num_1_lr_9e-05_wd_0.0001_max_words_64_rem_sw_True_lemm_False.pt'
)

_CONV_MODEL_PATH = os.path.join(
    _PROJECT_PATH, 'models', 'conv_net', 'saved_models',
    'FragConvNet_embed_200_filters_num_64_kern_[5, 10, 15]_drop_0.4_lr_0.0003_wd_0.0003_max_words_64_rem_sw_True_'
    'lemm_False_10-21-2020_12.49.59.pt'
)


def perform_prediction():
    mlp_model = get_fragmentized_mlp_model()
    lstm_model = get_lstm_model()
    cnn_model = get_cnn_model()

    mlp_result = predict_emotion(mlp_model, _LYRICS)
    lstm_result = predict_emotion(lstm_model, _LYRICS)
    cnn_result = predict_emotion(cnn_model, _LYRICS)

    print(f'Lyrics: {_LYRICS}')
    print(f'Predicted emotion:)'
          f'\n - mlp: {mlp_result}'
          f'\n - lstm: {lstm_result}'
          f'\n - cnn: {cnn_result}'
          )


def predict_emotion(model: BaseModel, lyrics: str) -> str:
    result = model.predict(lyrics)
    if result is not None:
        encoded_label, probabilities = result
        label = label_encoder.inverse_transform(encoded_label)
        return label[0]
    else:
        return 'Prediction did not succeed'


def get_fragmentized_mlp_model() -> FragmentizedMLPClassifier:
    mlp_model = FragmentizedMLPClassifier(input_size=200, output_size=4, dropout=0.5, weight_decay=5e-3,
                                          batch_size=128, removing_stop_words=True, lemmatization=False)
    mlp_model.load_state_dict(torch.load(_MLP_MODEL_PATH, map_location=_DEVICE))
    mlp_model.eval()
    return mlp_model


def get_lstm_model() -> FragmentizedLSTMClassifier:
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
    lstm_model.load_state_dict(torch.load(_LSTM_MODEL_PATH, map_location=_DEVICE))
    lstm_model.eval()
    return lstm_model


def get_cnn_model() -> FragmentizedConvNetClassifier:
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
    conv_model.load_state_dict(torch.load(_CONV_MODEL_PATH, map_location=_DEVICE))
    conv_model.eval()
    return conv_model


if __name__ == '__main__':
    perform_prediction()
