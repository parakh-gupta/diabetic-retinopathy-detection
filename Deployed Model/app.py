from __future__ import division, print_function
import os
import numpy as np
from keras.models import load_model

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import cv2

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

def int_to_classes(i):
    if i == 0: return "No DR"
    elif i == 1: return "Mild"
    elif i == 2: return "Moderate"
    elif i == 3: return "Severe"
    elif i == 4: return "Proliferative DR"
    print("Invalid class ", i)
    return "Invalid Class"

def preprocess(image):
    sigmaX=10
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
    return image

def model_predict(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = preprocess(img)
    arr = np.array(img, dtype="float") / 255.0
    preds = model.predict(arr.reshape(1,224,224,3))
    return preds


@app.route('/', methods=['GET'])
def index():
 	
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        # Process your result for human
        pred_class = np.argmax(preds, axis=-1)
        result = int_to_classes(pred_class)
        return result
    return None

if __name__ == '__main__':
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
    app.run(debug=True)
