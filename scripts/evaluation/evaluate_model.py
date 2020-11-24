import os
from typing import Dict

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

_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate_model(base_model: BaseModel) -> None:
    y_pred, y_test = base_model.test_model()

    print(classification_report(y_test, y_pred, target_names=_CLASS_NAMES))

    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=_CLASS_NAMES, columns=_CLASS_NAMES)
    show_confusion_matrix(df_cm)


def get_classification_report_for_evaluation(base_model: BaseModel) -> Dict:
    y_pred, y_test = base_model.test_model()
    report = classification_report(y_test, y_pred, target_names=_CLASS_NAMES, output_dict=True)
    return report


def show_confusion_matrix(conf_matrix: pd.DataFrame) -> None:
    hmap = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                       annot_kws={"fontsize": 18}, square=True)
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, fontsize=18)
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, fontsize=18)
    plt.ylabel('Rzecziwista klasa', fontsize=18)
    plt.xlabel('Predykowana klasa', fontsize=18)
    plt.tight_layout()
    plt.show()


_MLP_MODEL_PATH = os.path.join(
    _PROJECT_PATH, 'models', 'mlp', 'saved_models',
    'MLP_input_200_drop_0.5_lr_0.001_wd_1e-05_rem_sw_True_lemm_False_smote_False.pt'
)


def evaluate_mlp():
    print('MLP model')
    mlp_model = MLPClassifier(input_size=200, output_size=4, dropout=0.5, weight_decay=5e-3,
                              batch_size=128, removing_stop_words=True, lemmatization=False)
    mlp_model.load_state_dict(torch.load(_MLP_MODEL_PATH, map_location=device))
    evaluate_model(mlp_model)


_LSTM_MODEL_PATH = os.path.join(
    _PROJECT_PATH, 'models', 'lstm', 'saved_models',
    'LSTM_input_200_hidden_200_drop_0.3_lay_num_1_lr_0.0001_wd_0.0001_max_words_200_rem_sw_True_lemm_False.pt'
)


def evaluate_lstm():
    print('LSTM model')
    lstm_model = LSTMClassifier(
        input_dim=200,
        hidden_dim=200,
        output_dim=4,
        layer_dim=1,
        bidirectional=False,
        dropout=0.0,
        batch_size=128,
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
    'GRU_input_200_hidden_200_drop_0.3_lay_num_1_lr_9e-05_wd_1e-05_max_words_200_rem_sw_True_lemm_False.pt'
)


def evaluate_gru():
    print('GRU model')
    gru_model = GRUClassifier(
        input_dim=200,
        hidden_dim=200,
        output_dim=4,
        layer_dim=1,
        dropout=0.0,
        batch_size=128,
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
    'ConvNet_embed_200_filters_num_256_kern_[5, 10, 15]_drop_0.4_lr_0.0002_wd_0.0003_max_words_256_rem_sw_True_lemm_False_smote_False_v2.pt'
)


def evaluate_conv_net():
    print('ConvNet model')
    conv_model = ConvNetClassifier(
        embedding_dim=200,
        output_dim=4,
        dropout=0.5,
        batch_size=128,
        learning_rate=1e-5,
        weight_decay=65e-4,
        filters_number=256,
        kernels_sizes=[5, 10, 15],
        max_num_words=256,
        removing_stop_words=True,
        lemmatization=False
    )
    conv_model.load_state_dict(torch.load(_CONV_MODEL_PATH, map_location=device))
    evaluate_model(conv_model)


if __name__ == '__main__':
    evaluate_mlp()
    evaluate_lstm()
    evaluate_gru()
    evaluate_conv_net()
