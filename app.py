from flask import Flask, render_template, request, redirect
import os
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

# Set the upload folder
UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#load the cnn model
model = load_model('my_cnn2.h5')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return redirect(request.url)

    # Save the file to the upload folder
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        class_name = classification(filename)
        return render_template("display.html", filename=file.filename, name = class_name)


def classification(image_path):
    img = preprocessing.image.load_img(image_path, target_size=(100, 100))
    img_arr = preprocessing.image.img_to_array(img)
    img_cl = img_arr.reshape(1, 100, 100, 3)
    score = model.predict(img_cl)
    predicted_class = np.argmax(score)
    class_list = ['Otter', 'Sea Urchins', 'Seal','Sharks', 'Turtle_Tortoise']
    return class_list[predicted_class]



if __name__ == '__main__':
    app.run(debug=True)