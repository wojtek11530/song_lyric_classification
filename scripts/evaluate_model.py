import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from models.base import BaseModel
from models.conv_net.conv_net_model import ConvNetClassifier
from models.gru.gru_model import GRUClassifier
from models.label_encoder import label_encoder
from models.lstm.lstm_model import LSTMClassifier
from models.mlp.mlp_model import MLPClassifier

_CLASS_NAMES = label_encoder.classes_

_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate_model(base_model: BaseModel) -> None:
    y_pred, y_test = base_model.test_model()

    print(classification_report(y_test, y_pred, target_names=_CLASS_NAMES))

    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=_CLASS_NAMES, columns=_CLASS_NAMES)
    show_confusion_matrix(df_cm)


def show_confusion_matrix(conf_matrix: pd.DataFrame) -> None:
    hmap = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted sentiment')
    plt.tight_layout()
    plt.show()


_MLP_MODEL_PATH = os.path.join(
    _PROJECT_PATH, 'models', 'mlp', 'saved_models',
    'MLP_input_200_drop_0.5_lr_0.001_wd_1e-05_rem_sw_False_lemm_False_10-04-2020_14.15.28.pt'
)


def evaluate_mlp():
    mlp_model = MLPClassifier(input_size=200, output_size=4, dropout=0.5, weight_decay=5e-3,
                              batch_size=128, removing_stop_words=False, lemmatization=False)
    mlp_model.load_state_dict(torch.load(_MLP_MODEL_PATH, map_location=device))
    evaluate_model(mlp_model)


_LSTM_MODEL_PATH = os.path.join(
    _PROJECT_PATH, 'models', 'lstm', 'saved_models',
    'LSTM_input_200_hidden_200_drop_0.0_lay_num_1_lr_0.0001_wd_0.001_max_words_200_rem_sw_True_lemm_False_10-05-2020_15.23.34.pt'
)


def evaluate_lstm():
    lstm_model = LSTMClassifier(
        input_dim=200,
        hidden_dim=200,
        output_dim=4,
        layer_dim=1,
        bidirectional=False,
        dropout=0.0,
        batch_size=16,
        learning_rate=5e-3,
        weight_decay=5e-3,
        max_num_words=200,
        removing_stop_words=True,
        lemmatization=False
    )
    lstm_model.load_state_dict(torch.load(_LSTM_MODEL_PATH, map_location=device))
    evaluate_model(lstm_model)


_GRU_MODEL_PATH = os.path.join(
    _PROJECT_PATH, 'models', 'gru', 'saved_models',
    'GRU_input_200_hidden_200_drop_0.0_lay_num_1_lr_0.0001_wd_0.001_max_words_200_rem_sw_True_lemm_False_10-05-2020_15.03.23.pt'
)


def evaluate_gru():
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
    gru_model.load_state_dict(torch.load(_GRU_MODEL_PATH, map_location=device))
    evaluate_model(gru_model)


_CONV_MODEL_PATH = os.path.join(
    _PROJECT_PATH, 'models', 'conv_net', 'saved_models',
    'ConvNet_embed_200_filters_num_64_kern_[5, 10, 15]_drop_0.5_lr_0.0001_wd_0.0065_max_words_256_rem_sw_True_lemm_False_10-05-2020_14.09.03.pt'
)


def evaluate_conv_net():
    conv_model = ConvNetClassifier(
        embedding_dim=200,
        output_dim=4,
        dropout=0.5,
        batch_size=128,
        learning_rate=1e-4,
        weight_decay=5e-3,
        filters_number=64,
        kernels_sizes=[5, 10, 15],
        max_num_words=256,
        removing_stop_words=True,
        lemmatization=False
    )
    conv_model.load_state_dict(torch.load(_CONV_MODEL_PATH, map_location=device))
    evaluate_model(conv_model)


# evaluate_mlp()
evaluate_lstm()
evaluate_gru()
# evaluate_conv_net()
