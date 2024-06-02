from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import os
import cv2
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sklearn
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image
from keras.preprocessing import image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get the absolute path to the static folder
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

# Set the UPLOAD_FOLDER within the static folder
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'images')

# Ensure the UPLOAD_FOLDER exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print(UPLOAD_FOLDER)



# Function to extract HOG features from an image
def extract_hog_features(image):
    win_size = (64, 64)
    cell_size = (8, 8)
    block_size = (16, 16)
    nbins = 9

    hog = cv2.HOGDescriptor(win_size, block_size, cell_size, cell_size, nbins)
    features = hog.compute(image)
    features = features.flatten()
    return features

# Function to remove simple background (thresholding-based)
def remove_background(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a mask for the foreground
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=mask)

    return result

# Route to render the contact page
@app.route('/')
def default_route():
    return render_template('index.html')

@app.route('/ml')
def ml_route():
    return render_template('ml.html')

@app.route('/diseases')
def disease():
    return render_template('disease.html')

@app.route('/dl')
def index():
    return render_template('dl.html')
@app.route('/treatments')
def treatments():
    return render_template('treatment.html')
    
@app.route('/predict',methods = ['POST'])
def predict():
    file = request.files['image']
    filename = os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
    file.save(filename)

    predicted_disease, treatment = predict_disease_dl(filename)

    return jsonify({'disease': predicted_disease, 'treatment': treatment})

@app.route('/predict1',methods = ['POST'])
def predict1():
    file = request.files['image']
    filename = os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
    file.save(filename)

    predicted_disease, treatment = predict_disease_ml(filename)

    return jsonify({'disease': predicted_disease, 'treatment': treatment})


def predict_disease_dl(image_path):
    #Load the trained model
    model_filename = "prawn_disease_model.h5"
    model_path = os.path.join(os.getcwd(), "Models", model_filename)
    model = load_model(model_path)

    # Load and preprocess the input image
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    # Make predictions
    predictions = model.predict(img_array)

    # Get the class with the highest probability
    predicted_class = np.argmax(predictions)

    # Map class index to class label and treatment
    class_mapping = {
        0: ('BLACK GILL', "Remediating black gill disease in prawns involves maintaining water quality, implementing biosecurity measures, selective breeding for disease resistance traits, and using probiotics or immunostimulants to enhance immune response. Additionally, targeted treatments such as chemotherapy and environmental management practices can help mitigate outbreaks and reduce the spread of the disease."),
        1: ('HEALTHY', "No Remediation required."),
        2: ('VIBRIOSIS', "Remediating Vibriosis in prawns involves strict biosecurity protocols, such as quarantine measures and disinfection. Prophylactic measures like water quality management and probiotics use can bolster immune response. Targeted treatments with antibiotics or specific medications may be necessary during outbreaks, alongside environmental manipulation to reduce pathogen proliferation. Surveillance and selective breeding for disease resistance traits are also crucial for long-term management."),
        3: ('WHITE SPOT', "Remediating white spot disease in prawns involves strict biosecurity measures, disinfection protocols, selective breeding for disease resistance, and the use of probiotics or immunostimulants to bolster immune response. Additionally, minimizing stressors and implementing proper pond management practices are crucial for preventing outbreaks and controlling the spread of the disease.")
    }

    predicted_label, treatment = class_mapping.get(predicted_class, ('Unknown', 'Unknown'))
    print(f"The predicted class is: {predicted_label}")
    return predicted_label, treatment


def predict_disease_ml(image_path):
    # Load the model
    model_filename = "single_model.pkl"
    model_path = os.path.join(os.getcwd(), "Models", model_filename)
    model = joblib.load(model_path)

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    # Remove background
    img_no_background = remove_background(img)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img_no_background, cv2.COLOR_BGR2GRAY)

    # Resize the image
    img_resized = cv2.resize(img_gray, (64, 64))

    features = extract_hog_features(img_resized)

    # Reshape the features array to match the input format expected by the model
    features = features.reshape(1, -1)

    # Predict the label for the input image
    prediction = model.predict(features)[0]

    # Map the numerical label to the corresponding disease name and treatment
    disease_mapping = {
        0: ('BLACK GILL', "Remediating black gill disease in prawns involves maintaining water quality, implementing biosecurity measures, selective breeding for disease resistance traits, and using probiotics or immunostimulants to enhance immune response. Additionally, targeted treatments such as chemotherapy and environmental management practices can help mitigate outbreaks and reduce the spread of the disease."),
        1: ('VIBRIOSIS', "Remediating Vibriosis in prawns involves strict biosecurity protocols, such as quarantine measures and disinfection. Prophylactic measures like water quality management and probiotics use can bolster immune response. Targeted treatments with antibiotics or specific medications may be necessary during outbreaks, alongside environmental manipulation to reduce pathogen proliferation. Surveillance and selective breeding for disease resistance traits are also crucial for long-term management."),
        2: ('WHITESPOT', "Remediating white spot disease in prawns involves strict biosecurity measures, disinfection protocols, selective breeding for disease resistance, and the use of probiotics or immunostimulants to bolster immune response. Additionally, minimizing stressors and implementing proper pond management practices are crucial for preventing outbreaks and controlling the spread of the disease."),
        3: ('HEALTHY', "No  Remediation required.")
    }

    predicted_disease, treatment = disease_mapping.get(prediction, ('Unknown', 'Unknown'))

    return predicted_disease, treatment

if __name__ == '__main__':
    app.run(debug = True)







