from flask import Flask, request, render_template
import numpy as np
import os
from tensorflow import keras
from PIL import Image
from werkzeug.utils import secure_filename

# Initialize the Flask app and load the trained model
app = Flask(__name__)
model = keras.models.load_model("flowers17_model.h5")

# Replace with actual flower names (class labels)
class_labels = [
    "Daisy", "Dandelion", "Rose", "Sunflower", "Tulip",
    "Lily", "Orchid", "Violet", "Chrysanthemum", "Magnolia",
    "Lotus", "Cactus", "Bluebell", "Crocus", "Iris",
    "Poppy", "Marigold"
]

# Set up image upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploaded_images/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to check allowed extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Index route for uploading images
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        image = Image.open(filepath)
        image = image.resize((64, 64))
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array / 255.0

        # Predict class
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class]

        return render_template('index.html', prediction=predicted_class_label, image_url=filepath)

    return 'Invalid file format. Please upload a .jpg, .jpeg, or .png file.'

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
