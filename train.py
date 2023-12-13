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

img_augmentation = keras.models.Sequential(
    [
        keras.layers.experimental.preprocessing.RandomRotation(factor=0.15),
        keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.1,width_factor=0.1),
        keras.layers.experimental.preprocessing.RandomFlip(),
        keras.layers.experimental.preprocessing.RandomZoom(.5, .2),
    ],
    name="img_augmentation"
)

train_ds = prepare_for_training(train_ds,augment=True,shuffle=True)
val_ds = prepare_for_training(val_ds,cache=False,shuffle=False)
test_ds = prepare_for_training(test_ds,cache=False,shuffle=False)

METRICS = [
    'accuracy',
    tfa.metrics.F1Score(num_classes=len(CLASSES)),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall")
]

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

epochs = 25
keras.backend.clear_session()
model = create_model()
mcp_save = keras.callbacks.ModelCheckpoint('D:/Skripsi/Final Skripsi/Model/EfficientNet-B1 Skripsi epoch 100.h5', 
                                           save_best_only=True,
                                           monitor='val_accuracy')
history = model.fit(train_ds,
                    epochs=epochs,
                    validation_data=val_ds,
                    callbacks=[mcp_save])