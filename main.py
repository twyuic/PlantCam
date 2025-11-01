import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from predict import predict_image
from data import class_names
from gradcam import make_gradcam_heatmap, display_gradcam


'''gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU 已啟用:", gpus[0])
    except RuntimeError as e:
        print(e)
else:
    print("沒有偵測到 GPU，將使用 CPU")'''


model = load_model("plant_model.h5")


img_path = "sickpotato.jpg"  
img = load_img(img_path, target_size=(224,224))
img_array = img_to_array(img)/255.0
img_array = tf.expand_dims(img_array, 0)


pred = predict_image(img_path, class_names)
print("預測結果:", pred)


heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name='Conv_1')
display_gradcam(img_array[0], heatmap)
