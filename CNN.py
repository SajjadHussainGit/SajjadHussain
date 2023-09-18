# Libraries

#librraries used if this project
import os
import glob #for search of specific files
import random #use to generate randon model or intetergr through functions
import cv2
import imghdr #This module determines the type of image contained in a file or byte stream.
from shutil import copyfile #used to copy the content of the source file to the destination file
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy, sparse_categorical_crossentropy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # use to plot the images
from matplotlib.pyplot import imshow
%matplotlib inline

from keras.preprocessing.image import ImageDataGenerator # input data and transform into output at random basis
from tensorflow.keras.utils import load_img, img_to_array # convert image to numpy array
from keras import layers
from tensorflow.keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, AveragePooling2D
from keras import backend as K
from tensorflow.keras.layers import Input

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau #reduce the learning rate when a metric has stopped improving
from tensorflow.keras.callbacks import ModelCheckpoint # to saving model weight for later use
from tensorflow.keras.optimizers import SGD,Adam #  adjusting the weights in hidden layers
#Models and Layers
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

#accesing or locating path of folder in my device
os.path.join(r"C:\Users\DeLL\OneDrive\Desktop\Parkinsons\Data", 'no', 'yes')

#Checking list of folders
os.listdir(r"C:\Users\DeLL\OneDrive\Desktop\Parkinsons\Data")

# Avoid Out od memory Error by setting GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental_growth(gpu, 'True')

#Data set
data_dir = r"C:\Users\DeLL\OneDrive\Desktop\Parkinsons\Data"

#accepting image of below formate
image_exts = ['jpeg', 'jpg', 'png', 'bmp']

#joining path of yes and print image of Parkinson
os.listdir(os.path.join(data_dir, 'yes'))

#printing and resizing image 
img = cv2.imread(os.path.join(r"C:\Users\DeLL\OneDrive\Desktop\Parkinsons\Data", 'no', '2 no.jpeg'))
new_img = cv2.resize(img, (256, 256))
plt.imshow(img)
plt.show()

#printing clasess of image
for image_class in os.listdir(data_dir):
    print(image_class)

#details of tensor API
tf.keras.utils.image_dataset_from_directory??

#Converting data into batch size, and reducing image size etc
data  = tf.keras.utils.image_dataset_from_directory(r"C:\Users\DeLL\OneDrive\Desktop\Parkinsons\Data")

#execute the consective batche of images
data_iterator = data.as_numpy_iterator()

#printing shape of batch and generating random batch of 32 images where o represent images and 1 represent labels
batch = data_iterator.next()
batch[0].shape

# lables
# Class 1 = parkinson 
# Class 0 = Non_parkinson
batch[1]

#Printing random images from batch with labels
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])

#rescaling image between 0 and 1
scaled = batch[0]/255

#checking max and minmum number. minimum number must be 0 and maximum 1
scaled.min(), scaled.max()

# Preprocessing of Data

#extract the data through pipline thats why we can quiackly access oure data, x is images and y is labels
#lambda is maping function of input x and output y
data = data.map(lambda x, y: (x/255, y))
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()
batch[0].max()

#checking again image size and label of image are working correctly
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])

# Spliting Data into training and testing 

# Checking total batches of data, Total 7 batches and each bathes has 32 images
#printing length of data
len(data)

#Spliting Data into Training, validation and testing
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)+1
test_size = int(len(data)*.1)+1
train_size+val_size+test_size

#Creating Variable of training, validation and testing
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)
len(test)

# Customised CNN Model

# Using API sequential Models because we need output in sequence, 
#Building Model

model = Sequential([
    
    layers.Conv2D(filters=32, kernel_size=(5,5), activation="relu", input_shape=[256, 256, 3]),
    layers.MaxPool2D(),
    
    layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu"),
    layers.MaxPool2D(),
    
    layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu"),
    layers.MaxPool2D(),
    
    layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu"),
    layers.MaxPool2D(),
    
    layers.Flatten(),
    layers.Dropout(.25),
    layers.Dense(units=256, activation="relu"),
    layers.Dense(units=2, activation="softmax"),
])
model.summary()

#Compiling of Model
model.compile(optimizer=tf.keras.optimizers.Adam(epsilon=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Training of Models

# providing the measurements and visualizations needed during the machine learning workflow
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
#Tensorboard create this to save date to visulize
logdir = r"C:\Users\DeLL\OneDrive\Desktop\Parkinsons\Logs"
# infuture to check model performance and dropping learning rate
#including early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=0.001, restore_best_weights=True,)
hist = model.fit(train, epochs=10, validation_data = val, callbacks = [tensorboard_callback])

# hist.history

# Plot performance

#plot of loss and val_loss
fig = plt.figure()
plt.plot(hist.history['loss'], color = 'green', label = 'Training_loss')
plt.plot(hist.history['val_loss'], color = 'red', label = 'Validation_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc = 'upper left')
plt.show()

# plot of accuracy
fig = plt.figure()
plt.plot(hist.history['accuracy'], color = 'orange', label = 'Training_accuracy')
plt.plot(hist.history['val_accuracy'], color = 'blue', label = 'Validation_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc = 'upper left')
plt.show()

#Evaluate performance

#Creating variable for precision, recall, and binary accuracy
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

#Checking length of test folder
len(test)

#Testing of Model
img = cv2.imread(r"C:\Users\DeLL\OneDrive\Desktop\Parkinsons\MRI\18 no.jpg")
plt.imshow(img)
plt.show()

#Resizing image for testing
resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()

resize.shape

np.expand_dims(resize, 0).shape

yhat = model.predict(np.expand_dims(resize/255, 0))
yhat

#if yhatnew>0.5:
    #print('Predicted class is Parkinson')
#else:
    #print('Predicted class is not Parkinson')

#Saving of Models
model.save(os.path.join(r"C:\Users\DeLL\OneDrive\Desktop\Parkinsons\Models", 'Parkinson_pred.h5'))
os.path.join(r"C:\Users\DeLL\OneDrive\Desktop\Parkinsons\Models", 'Parkinson_pred.h5')

#Extracting model for checking image
new_model = load_model(os.path.join(r"C:\Users\DeLL\OneDrive\Desktop\Parkinsons\Models", 'Parkinson_pred.h5'))
new_model

#new variable for saved model
yhatnew = new_model.predict(np.expand_dims(resize/255, 0))

#if yhatnew>0.5:
    #print('Predicted class is Parkinson')
#else:
    #print('Predicted class is not Parkinson')

# CNN VGG16

trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory=r"C:\Users\DeLL\OneDrive\Desktop\Parkinsons\Train_Data",target_size=(256,256))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory=r"C:\Users\DeLL\OneDrive\Desktop\Parkinsons\Test_Data", target_size=(256,256))

vggmodel = Sequential()
vggmodel.add(Conv2D(input_shape=(256,256,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
vggmodel.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
vggmodel.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
vggmodel.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
vggmodel.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
vggmodel.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
vggmodel.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
vggmodel.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
vggmodel.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
vggmodel.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
vggmodel.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
vggmodel.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
vggmodel.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
vggmodel.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
vggmodel.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
vggmodel.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
vggmodel.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
vggmodel.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
vggmodel.add(Flatten())
vggmodel.add(Dense(units=4096,activation="relu"))
vggmodel.add(Dense(units=4096,activation="relu"))
vggmodel.add(Dense(units=1, activation="softmax"))
#Compiling of Model
vggmodel.compile(optimizer=tf.keras.optimizers.Adam(epsilon=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
vggmodel.summary()

logdir = r"C:\Users\DeLL\OneDrive\Desktop\Parkinsons\Logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir) 
# infuture to check model performance and dropping learning rate
#including early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=0.001, restore_best_weights=True,)
hist = model.fit(train, epochs=10, validation_data = val, callbacks = [tensorboard_callback])

fig = plt.figure()
plt.plot(hist.history['loss'], color = 'teal', label = 'Training_loss')
plt.plot(hist.history['val_loss'], color = 'orange', label = 'Validation_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc = 'upper left')
plt.show()

# plot of accuracy
fig = plt.figure()
plt.plot(hist.history['accuracy'], color = 'red', label = 'Training_accuracy')
plt.plot(hist.history['val_accuracy'], color = 'blue', label = 'Validation_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc = 'upper left')
plt.show()

