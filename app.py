from flask import Flask, request, render_template

from src.predict import predict
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/process_audio', methods=['POST'])
def process_audio():
    file = request.files['audio_file']
    detected_genre = predict(file)
    print('genre: ', detected_genre)
    confidence = 0.85
    return render_template(
        'result.html',
        genre=detected_genre,
        confidence=confidence
    )


if __name__ == '__main__':
    app.run()
