from typing import Dict, Optional

from flask import Flask, abort, jsonify, request
from flask_cors import CORS

from backend_app.results import add_to_results, get_average_results
from backend_app.used_model import get_model
from models.label_encoder import label_encoder

app = Flask(__name__)
app.config["DEBUG"] = False
CORS(app)

_CLASS_NAMES = label_encoder.classes_

_model = get_model()


@app.route('/', methods=['GET'])
def hello():
    return "<h1>Lyric Emotion Backend</h1><p>Server is working.</p>"


@app.route('/song_emotion', methods=['POST'])
def get_song_emotion():
    if not request.json or 'lyrics' not in request.json:
        abort(400)

    lyrics = request.json['lyrics']
    title = request.json['artist']
    artist = request.json['title']

    emotion_probabilities = _get_emotion_probabilities(lyrics)

    if emotion_probabilities is None:
        return '', 204
    else:
        add_to_results(title, artist, emotion_probabilities)
        average_results = get_average_results()
        return jsonify([emotion_probabilities, average_results]), 200


def _get_emotion_probabilities(lyrics: str) -> Optional[Dict[str, float]]:
    result = _model.predict(lyrics)
    if result is not None:
        _, encoded_label_probabilities = result
        emotion_probabilities = {}
        for encoded_label, prob in enumerate(encoded_label_probabilities):
            label = label_encoder.inverse_transform([encoded_label])[0]
            emotion_probabilities[label] = float(prob)
        return emotion_probabilities
    else:
        return None
