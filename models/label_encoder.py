from sklearn.preprocessing import LabelEncoder

_SENTIMENT_VALUES = ['angry', 'happy', 'relaxed', 'sad']

label_encoder = LabelEncoder().fit(_SENTIMENT_VALUES)
