from keras import models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = models.load_model("model.h5")
def preprocess_image(image_path, target_size=(225, 225)):
    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    return x
x = preprocess_image('D:/dsproject/vit/test/test/download.jpg')
predictions = model.predict(x)
predictions[0]
labels = {0: 'Apple___Apple_scab', 1: 'Apple___Black_rot', 2: 'Apple___healthy'}
predicted_label = labels[np.argmax(predictions)]
print(predicted_label)