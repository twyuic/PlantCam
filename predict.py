from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from data import preprocess

model = load_model('plant_model.h5')

def predict_image(img_path, class_names):
    img = tf.keras.utils.load_img(img_path, target_size=(224,224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    pred = model.predict(img_array)
    pred_class = np.argmax(pred, axis=1)[0]
    return class_names[pred_class]


if __name__ == "__main__":
    img_path = "sickpotato.jpg"
    from data import load_data
    _, _, _, class_names = load_data()
    pred = predict_image(img_path, class_names)
    print("預測結果:", pred)