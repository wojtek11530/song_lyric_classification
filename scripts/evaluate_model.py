import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from models.base import BaseModel
from models.label_encoder import label_encoder
from models.lstm.lstm_model import LSTMClassifier
from models.mlp.mlp_model import MLPClassifier

_CLASS_NAMES = label_encoder.classes_

_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LSTM_MODEL_PATH = os.path.join(_PROJECT_PATH, 'models', 'lstm', 'saved_lstm_model_09-26-2020_14.46.17.pt')

_MLP_MODEL_PATH = os.path.join(_PROJECT_PATH, 'models', 'mlp', 'saved_mlp_model_09-26-2020_15.22.41.pt')
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


lstm_model = LSTMClassifier(
    input_dim=200,
    output_dim=4,
    bidirectional=False,
    dropout=0.5,
    batch_size=32,
    layer_dim=1,
    learning_rate=1e-4,
    weight_decay=5e-3
)
lstm_model.load_state_dict(torch.load(_LSTM_MODEL_PATH, map_location=device))
evaluate_model(lstm_model)

mlp_model = MLPClassifier(input_size=200, output_size=4, dropout=0.5, weight_decay=5e-3, batch_size=64)
mlp_model.load_state_dict(torch.load(_MLP_MODEL_PATH, map_location=device))
evaluate_model(mlp_model)
