from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Define the image size and categories
img_height, img_width = 224, 224
categories = ['Aloevera', 'Lemon', 'Lemongrass', 'Maka', 'Turmeric']

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Define upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define allowed extensions for file uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to predict image class
def predict_image_class(image_path):
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    print(f'Image array shape: {img_array.shape}')
    
    predictions = model.predict(img_array)
    
    print(f'Predictions: {predictions}')
    
    predicted_class = categories[np.argmax(predictions)]
    
    return predicted_class

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Predict the class of the uploaded image
            predicted_class = predict_image_class(filepath)
            return render_template('result.html', predicted_class=predicted_class)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
