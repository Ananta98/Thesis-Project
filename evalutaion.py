import time
import os
import random
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
import matplotlib.pyplot as plt
import efficientnet.tfkeras as enet
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

IMG_SIZE = 240
data_dir = 'D:/Skripsi/Final Skripsi/COVID-19 Radiography Database'

covid_filenames = tf.io.gfile.glob(str(data_dir + '/COVID/*'))
normal_filenames = tf.io.gfile.glob(str(data_dir + '/NORMAL/*'))
pneumonia_filenames = tf.io.gfile.glob(str(data_dir + '/Viral Pneumonia/*'))

CLASSES = ['NORMAL', 'COVID', 'Viral Pneumonia']
covid_train_filenames, covid_test_filenames = train_test_split(covid_filenames, test_size=0.2,shuffle=False)
covid_train_filenames, covid_val_filenames = train_test_split(covid_train_filenames, test_size=0.1,shuffle=False)

normal_train_filenames, normal_test_filenames = train_test_split(normal_filenames, test_size=0.2, shuffle=False)
normal_train_filenames, normal_val_filenames = train_test_split(normal_train_filenames, test_size=0.1, shuffle=False)

pneumonia_train_filenames, pneumonia_test_filenames = train_test_split(pneumonia_filenames, test_size=0.2, shuffle=False)
pneumonia_train_filenames, pneumonia_val_filenames = train_test_split(pneumonia_train_filenames, test_size=0.1, shuffle=False)

train_filenames = []
train_filenames.extend(covid_train_filenames)
train_filenames.extend(normal_train_filenames)
train_filenames.extend(pneumonia_train_filenames)

val_filenames = []
val_filenames.extend(covid_val_filenames)
val_filenames.extend(normal_val_filenames)
val_filenames.extend(pneumonia_val_filenames)

test_filenames = []
test_filenames.extend(covid_test_filenames)
test_filenames.extend(normal_test_filenames)
test_filenames.extend(pneumonia_test_filenames)

train_list_ds = tf.data.Dataset.from_tensor_slices(train_filenames)
val_list_ds = tf.data.Dataset.from_tensor_slices(val_filenames)
test_list_ds = tf.data.Dataset.from_tensor_slices(test_filenames)

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] == CLASSES

def decode_img(img):
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, (IMG_SIZE,IMG_SIZE))

def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

def create_model():
    num_classes = 3
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    model = enet.EfficientNetB1(include_top=False,input_tensor=inputs, weights='noisy-student')
    outputs = keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    outputs = keras.layers.Dropout(0.3)(outputs)
    outputs = keras.layers.Dense(num_classes,activation="softmax",kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-3),name="pred")(outputs)
    outputs = keras.layers.Dense(num_classes,activation="softmax")(outputs)
    model = tf.keras.Model(inputs,outputs)
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=METRICS)
    return model

model = create_model()
model = keras.models.load_model('D:/Skripsi/Final Skripsi/Model/EfficientNet-B1 Skripsi epoch 100.h5')
model.evaluate(test_ds, return_dict=True)

start_time = time.time()
predicted_all_classes = model.predict(test_ds)
last_time = time.time()
print(f"execution time : {last_time - start_time}s")

y = np.concatenate([y for x, y in test_ds])
y_true = np.argmax(y,axis=1)
predicted = np.argmax(predicted_all_classes,axis=1)
matrix = confusion_matrix(y_true, predicted)
plot_confusion_matrix(matrix,CLASSES)