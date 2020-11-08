import flask

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def hello():
    return "<h1>Lyric Emotion Backend</h1><p>Server is working.</p>"
