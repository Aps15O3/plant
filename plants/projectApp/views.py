from django.shortcuts import render
from .forms import ImageForm
from keras import models
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import h5py
import numpy as np
import tensorflow as tf
import time
# Create your views here.
def home(request):
    model = tf.keras.models.load_model(os.path.abspath(os.getcwd()+"/model.h5"))
    print(os.path.exists(os.path.abspath(os.getcwd()+"/model.h5") ))
    def preprocess_image(image_path, target_size=(225, 225)):
        img = load_img(image_path, target_size=target_size)
        x = img_to_array(img)
        x = x.astype('float32') / 255.
        x = np.expand_dims(x, axis=0)
        return x
   
    print("3456")
    print(os.getcwd())
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # Get the current instance object to display in the template
            img_obj = form.instance
            time.sleep(2)
            print(os.path.exists(os.getcwd()+img_obj.image.url))
            def preprocess_image(image_path, target_size=(225, 225)):
                img = load_img(image_path, target_size=target_size)
                x = img_to_array(img)
                x = x.astype('float32') / 255.
                x = np.expand_dims(x, axis=0)
                return x
            x = preprocess_image(os.getcwd()+img_obj.image.url)
            predictions = model.predict(x)
            predictions[0]
            labels = {0: 'Apple___Apple_scab', 1: 'Apple___Black_rot', 2: 'Apple___healthy'}
            predicted_label = labels[np.argmax(predictions)]
            print(predicted_label)
            return render(request, 'home.html', {'form': form, 'img_obj': img_obj, 'result':predicted_label})
    else:
        form = ImageForm()    
        return render(request, 'home.html', {'form': form})
    