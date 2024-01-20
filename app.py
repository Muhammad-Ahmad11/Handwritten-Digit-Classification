from flask import Flask, render_template, request
from keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)
model = load_model('trained_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    img = request.files['image']
    
    # Read the image file and convert to grayscale
    img = cv2.imdecode(np.fromstring(img.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    
    # Resize the image to 28x28 pixels
    img = cv2.resize(img, (28, 28))
    
    # Preprocess the image
    img = img.reshape(1, 28, 28, 1).astype('float32')
    img /= 255
    
    # Make predictions
    predictions = model.predict(img)
    predicted_label = np.argmax(predictions)
    
    return str(predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
