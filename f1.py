import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, f1_score
from data import load_data, preprocess


BATCH_SIZE = 32


_, ds_test, num_classes, class_names = load_data()
AUTOTUNE = tf.data.AUTOTUNE
ds_test = ds_test.map(preprocess, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)


model = tf.keras.models.load_model("plant_model.h5")
print("模型載入完成！")


y_true = []
y_pred = []

for images, labels in ds_test:
    preds = model.predict(images)
    preds_class = np.argmax(preds, axis=1)
    y_pred.extend(preds_class)
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)


print("分類報告：")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

macro_f1 = f1_score(y_true, y_pred, average='macro')
print(f"Macro F1-score: {macro_f1:.4f}")
