from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = load_model('rice.h5')

class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
      
        filepath = os.path.join('static/uploads', file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

       
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100
        all_confidences = {class_names[i]: f"{predictions[0][i]*100:.2f}%" for i in range(len(class_names))}

                return render_template('results.html',
                               prediction=predicted_class,
                               confidence=f"{confidence:.2f}%",
                               all_confidences=all_confidences,
                               user_image=filepath)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
