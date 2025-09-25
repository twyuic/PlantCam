import tensorflow as tf
from data import load_data, preprocess, augment
from model import build_model


'''gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU 已啟用:", gpus[0])
    except RuntimeError as e:
        print(e)
else:
    print("沒有偵測到 GPU，將使用 CPU")'''


IMG_SIZE = 224
BATCH_SIZE = 16  
EPOCHS = 5       

ds_train, ds_test, num_classes, class_names = load_data()
AUTOTUNE = tf.data.AUTOTUNE
ds_train = ds_train.map(augment, num_parallel_calls=AUTOTUNE).shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
ds_test = ds_test.map(preprocess, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

ds_train, ds_test, num_classes, class_names = load_data()
print(class_names)


model = build_model(num_classes)


history = model.fit(
    ds_train,
    validation_data=ds_test,
    epochs=EPOCHS
)


model.save('plant_model.h5')
print("模型已存成 plant_model.h5")
