import os
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename


from genre_detector.predict import predict
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/genre', methods=['POST'])
def genre():
    file = request.files['audio_file']
    print(file)
    detected_genre = predict(file)
    print('genre: ', detected_genre)
    confidence = 85

    # save the audio file to a local directory
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    return render_template(
        'result.html',
        genre=detected_genre,
        confidence=confidence,
        audio_file=filename
    )


@app.route('/play_audio/<filename>')
def play_audio(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), mimetype='audio/mpeg')


if __name__ == '__main__':
    app.run()
