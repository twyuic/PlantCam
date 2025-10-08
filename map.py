
%cd /content/drive/MyDrive/plants





import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from data import load_data

IMG_SIZE = 224
img_path = "sickpotato.jpg"   
PATCH_SIZE = 8            
STRIDE = 4                  
ALPHA = 0.6                  

_, _, _, class_names = load_data()


model = load_model("plant_model.h5")
print("模型載入完成！")


img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
img_array = img_to_array(img)/255.0
img_batch = tf.expand_dims(img_array, 0)


pred = model.predict(img_batch)
pred_class = np.argmax(pred[0])
pred_name = class_names[pred_class]
pred_prob = pred[0][pred_class]
print(f"模型預測: {pred_name} (機率: {pred_prob:.2f})")


heatmap = np.zeros((IMG_SIZE, IMG_SIZE))

for i in range(0, IMG_SIZE, STRIDE):
    for j in range(0, IMG_SIZE, STRIDE):
        occluded_img = img_array.copy()
     
        occluded_img[i:i+PATCH_SIZE, j:j+PATCH_SIZE, :] = 0
        occluded_batch = tf.expand_dims(occluded_img, 0)
        occluded_pred = model.predict(occluded_batch)
        
        heatmap[i:i+PATCH_SIZE, j:j+PATCH_SIZE] = pred_prob - occluded_pred[0][pred_class]


heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

jet = plt.cm.get_cmap("jet")
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[(heatmap*255).astype(np.int32)]
jet_heatmap_img = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
jet_heatmap_img = jet_heatmap_img.resize((IMG_SIZE, IMG_SIZE))

superimposed_img = tf.keras.preprocessing.image.array_to_img(
    ALPHA * np.array(jet_heatmap_img) + np.array(img)
)

plt.imshow(superimposed_img)
plt.axis('off')
plt.title(f"Sensitivity Map: {pred_name}")
plt.show()
