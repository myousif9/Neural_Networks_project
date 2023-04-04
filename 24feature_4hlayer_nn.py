# -*- coding: utf-8 -*-
"""
Created on March 13, 2023

@author: Mauricio Cespedes Tenorio and Mohamed Yousif
"""
# Libraries
# from turtle import shape
# from Deblurring_NN import X_test
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
from keras.models import load_model
from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras import models
from keras import layers
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras import backend as K
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from random import randint
import zipfile
import cv2

print(sys.argv)


# Path to input file
inputZip_blur = str(sys.argv[1])
inputZip_origin = str(sys.argv[2])

# Path to output dir
saveDir = str(sys.argv[3]) # was called path
task =  str(sys.argv[4])

# Zip file to save test cases
outZip = os.path.join(saveDir, f"CNN_{task}_tests.zip")

# Parameters
kernel_s = 5
batch_size = 32
epochs = 100
imageSize = 128


# functions for extracting and loading train and test data
def extract_train(inputZip,imageSize):
    with zipfile.ZipFile(inputZip, mode="r") as archive:
        # Separate files into train, validation and test
        train_set, test_set = train_test_split(archive.namelist(), test_size=0.3, random_state=0)
        val_set, test_set = train_test_split(test_set, test_size=0.5, random_state=0)
        train = np.zeros([len(train_set), imageSize, imageSize, 3])
        val = np.zeros([len(val_set), imageSize, imageSize, 3])

        for idx, filename in enumerate(train_set):
            data = archive.read(filename)
            image = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
            train[idx,:,:,:] = image.astype('float32')/255.0

        for idx, filename in enumerate(val_set):
            data = archive.read(filename)
            image = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
            val[idx,:,:,:] = image.astype('float32')/255.0
            
    return train, test_set, val

def extract_test(inputZip,test_set):
    with zipfile.ZipFile(inputZip, mode="r") as archive:
        
        test = np.zeros([len(test_set), imageSize, imageSize, 3])

        for idx, filename in enumerate(test_set):
            data = archive.read(filename)
            image = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
            test[idx,:,:,:] = image.astype('float32')/255.0

    return test

# extracting training and testing data
X_train, X_test_set, X_val = extract_train(inputZip_blur, imageSize)

y_train, y_test_set, y_val = extract_train(inputZip_origin, imageSize)


# load illusions
illusions = np.load('output/Test/ilussions.npy')

print("Data loaded!!")

# defining neural net
model = models.Sequential()

model.add(Conv2D(24,(kernel_s, kernel_s), padding='same',activation='sigmoid',input_shape=(128, 128, 3)))
model.add(Conv2D(24,(kernel_s, kernel_s), padding='same',activation='sigmoid'))
model.add(Conv2D(24,(kernel_s, kernel_s), padding='same',activation='sigmoid'))
model.add(Conv2D(3,(kernel_s, kernel_s), padding='same',activation='sigmoid'))

model.compile(optimizer='adam', loss='mean_squared_error')

es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
chkpt = saveDir +f'{task}_model_simple.hdf5'
cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_val, y_val),
                    callbacks=[es_cb, cp_cb],
                    shuffle=True)
                    
del X_train, y_train, X_val, y_val

x_test = extract_test(inputZip_blur, X_test_set)
y_test = extract_test(inputZip_blur, y_test_set)

score = model.evaluate(x_test, y_test, verbose=1)    


## Compute PSNR and SSIM in test set
  
out = model.predict(x_test)*255  

# New zip file
zipf = zipfile.ZipFile(outZip, 'w', zipfile.ZIP_DEFLATED)
for idx in range(out.shape[0]):
    image = out[idx,:,:,:]
    retval, buf = cv2.imencode('.jpeg', image)
    filename = X_test_set[idx]
    zipf.writestr(filename, buf)

zipf.close()

outIllusions = model.predict(illusions)  

performance = np.zeros((x_test.shape[0],2))

# PSNR
for i in range(x_test.shape[0]):    
    
    performance[i,0] = psnr(y_test[i,:,:,:]*255,out[i,:,:,:],data_range=255)
    
# SSIM
for i in range(x_test.shape[0]):    
    
    performance[i,1] = ssim(y_test[i,:,:,:]*255, out[i,:,:,:], data_range=255, gaussian_weights=True
    , sigma = 1.5, use_sample_covariance=False,multichannel=True)


## Save results
file = open(os.path.join(saveDir,f'Illusions_{task}.txt'),'w')
file.write('Mean PSNR'+str(np.mean(performance[:,0]))+'-'+str(np.std(performance[:,0]))) 
file.write('Mean SSIM'+str(np.mean(performance[:,1]))+'-'+str(np.std(performance[:,1]))) 
file.write(chkpt)
file.close() 

print('Saving Results ...')
np.save(os.path.join(saveDir,f'Illusions_{task}'),outIllusions) 
                