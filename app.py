import os
from flask import Flask, render_template, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import face_recognizer as fr

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return render_template('index.html')

@app.route('/detail/<filename>')
def uploaded_file(filename):
    emotion = fr.get_emotion('static\\uploads\\' + filename)
    song = 'piano.wav'
    if emotion[0] == 'happiness':
        song = 'happy_generated.mid'
    if emotion[0] == 'sadness':
        song = 'sad_generated.mid'
    return render_template('detail.html', filename = filename, emotion = emotion, song = song)
    # return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    

if __name__ == '__main__':
    app.run()