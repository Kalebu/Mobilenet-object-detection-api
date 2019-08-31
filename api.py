from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, url_for, request 
from werkzeug import secure_filename
import os

app = Flask(__name__)

home = os.getcwd()


#___________loading keras model___________________
model = tf.keras.models.load_model('brain.h5', compile=False)




#_____________________Function to predict categories____________________
def predict_classes(imagename):
    try:

        image_path = 'static/'+imagename
        picture = image.load_img(image_path, target_size = (224, 224))
        picture_array = image.img_to_array(picture)
        picture_array = np.expand_dims(picture_array, axis=0)
        picture_array = preprocess_input(picture_array)

        prediction_result = model.predict(picture_array)

        decoded_result = decode_predictions(prediction_result, top=12)[0]

        labels = [y for x , y, z in decoded_result]

        return labels

    except Exception as ex:
        print(ex)
        print('The above error have occured')




#________Function to save image_______________________
def save_image(file, img):
    try:
        
        image_dir = 'static'
        os.chdir(image_dir)
        file.save(img)
        os.chdir(home)

        predicted = predict_classes(img)

        return predicted

    except Exception as ex:
        os.chdir(home)
        print(ex)


#______________React route________________________
@app.route('/', methods=['POST', 'GET'])
def react():
    if request.method == 'POST':
        
        file = request.files['pic'];

        name = secure_filename(file.filename)

        predicted = save_image(file, name)

        return predicted

    else:
        return 'Methods restricted'


if __name__ == '__main__':
    app.run(debug=True)
