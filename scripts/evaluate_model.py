import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from models.base import BaseModel
from models.conv_net.conv_net_model import ConvNetClassifier
from models.label_encoder import label_encoder
from models.lstm.lstm_model import LSTMClassifier
from models.mlp.mlp_model import MLPClassifier

_CLASS_NAMES = label_encoder.classes_

_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LSTM_MODEL_PATH = os.path.join(
    _PROJECT_PATH, 'models', 'lstm',
    'LSTM_input_200_drop_0.5_lay_num_1_lr_0.005_wd_0.005_max_words_200_rem_sw_True_v1.pt'
)
_MLP_MODEL_PATH = os.path.join(_PROJECT_PATH, 'models', 'mlp',
                               'MLP_input_200_drop_0.5_lr_0.001_wd_1e-05_09-27-2020_13.54.38.pt')

_CONV_MODEL_PATH = os.path.join(
    _PROJECT_PATH, 'models', 'conv_net',
    'ConvNet_embed_200_filters_num_32_kern_[3, 5, 7, 10, 15]_drop_0.5_lr_0.01_wd_0.001_max_words_256_rem_sw_False_09-29-2020_10.25.51.pt'
)

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


def evaluate_lstm():
    lstm_model = LSTMClassifier(
        input_dim=200,
        output_dim=4,
        bidirectional=False,
        dropout=0.5,
        batch_size=32,
        layer_dim=1,
        learning_rate=5e-3,
        weight_decay=5e-3,
        max_num_words=100,
        removing_stop_words=True
    )
    lstm_model.load_state_dict(torch.load(_LSTM_MODEL_PATH, map_location=device))
    evaluate_model(lstm_model)


def evaluate_mlp():
    mlp_model = MLPClassifier(input_size=200, output_size=4, dropout=0.5, weight_decay=5e-3,
                              batch_size=64, removing_stop_words=True)
    mlp_model.load_state_dict(torch.load(_MLP_MODEL_PATH, map_location=device))
    evaluate_model(mlp_model)


def evaluate_conv_net():
    conv_model = ConvNetClassifier(
        embedding_dim=200,
        output_dim=4,
        dropout=0.5,
        batch_size=32,
        learning_rate=1e-2,
        weight_decay=1e-3,
        filters_number=32,
        kernels_sizes=[3, 5, 7, 10, 15],
        max_num_words=256,
        removing_stop_words=False
    )
    conv_model.load_state_dict(torch.load(_CONV_MODEL_PATH, map_location=device))
    evaluate_model(conv_model)


# evaluate_mlp()
# evaluate_lstm()
evaluate_conv_net()
