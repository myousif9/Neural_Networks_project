# -*- coding: utf-8 -*-
"""
Created on March 13, 2023

@author: Mauricio Cespedes Tenorio
"""
# Libraries
from turtle import shape
from Deblurring_NN import X_test
import keras
from keras.models import load_model
from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras import models
from keras import layers
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import backend as K
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from random import randint
import zipfile
import cv2

# Image manipulation.
# import PIL.Image
# from io import BytesIO

# def load_image(filename):
#     #image = PIL.Image.open(filename)

#     image = PIL.Image.open(filename) # open colour image
#     # image = image.convert('L') # convert image to black and white
#     image = np.array(image)
    
#     return np.float32(image)

#sys.argv 

print(sys.argv)

# Path to input file
inputZip_blur = str(sys.argv[1])
inputZip_origin = str(sys.argv[2])

# Path to output dir
saveDir = str(sys.argv[3]) # was called path

# Zip file to save test cases
outZip = os.path.join(saveDir, "CNN_denoising_tests.zip")

# Parameters
kernel_s = 5
# poolSize = 2
# numKernels = int(sys.argv[4])
batch_size = 32
epochs = 100
imageSize = 128



def extract_train(inputZip,imageSize):
    with zipfile.ZipFile(inputZip, mode="r") as archive:
        # Separate files into train, validation and test
        train_set, test_set = train_test_split(archive.namelist(), test_size=0.3, random_state=0)
        val_set, test_set = train_test_split(test_set, test_size=0.5, random_state=0)
        train = np.zeros([len(archive.namelist())*0.3, imageSize, imageSize, 3])
        val = np.zeros([len(archive.namelist())*0.15, imageSize, imageSize, 3])

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
        
        test = np.zeros([len(archive.namelist())*0.15, imageSize, imageSize, 3])

        for idx, filename in enumerate(test_set):
            data = archive.read(filename)
            image = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
            test[idx,:,:,:] = image.astype('float32')/255.0

    return test

X_train, X_test_set, X_val = extract_train(inputZip_blur, imageSize)

y_train, y_test_set, y_val = extract_train(inputZip_origin, imageSize)



# Perform training using the zip file
# with zipfile.ZipFile(inputZip, mode="r") as archive:
#     # Separate files into train, validation and test
#     train_set, test_set = train_test_split(archive.namelist(), test_size=0.3, random_state=0)
#     val_set, test_set = train_test_split(test_set, test_size=0.3, random_state=0)
#     for filename in train_set:
#         data = archive.read(filename)
#         image = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
        
#         retval, buf = cv2.imencode('.jpeg', image)
#         zipf.writestr(filename, buf)
# zipf.close()

# inputImg = np.load(path+'output_blur.npy')
# outputImg = np.load(path+'output_blur.npy')
illusions = np.load('output/Test/ilussions.npy')

print("Data loaded!!")


# x = inputImg.astype('float32')
# y = outputImg.astype('float32')
# x_train /= 255
# x_test /= 255
# x /= 255
# y /= 255


# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=randint(0, 100))


# divide x_test into validation and test
# x_val = X_test[:2000]
# y_val = y_test[:2000]
# x_test = X_test[2000:]
# y_test = y_test[2000:]


# print("Data Normalized!!")

model = models.Sequential()

model.add(Conv2D(3,(kernel_s, kernel_s), padding='same',activation='sigmoid'),input_shape=(128, 128, 3))
model.add(Conv2D(3,(kernel_s, kernel_s), padding='same',activation='sigmoid'),input_shape=(128, 128, 3))

# input_img = Input(shape=(128, 128, 3))
# x = Conv2D(3, (kernel_s, kernel_s), padding='same')(input_img)
# x = Activation('sigmoid')(x)

# x = Conv2D(3, (kernel_s, kernel_s), padding='same')(x)
# decoded = Activation('sigmoid')(x)

# model = Model(input_img, decoded)
model.compile(optimizer='adam', loss='mean_squared_error')


es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
chkpt = saveDir +'deblurring_model_simple.hdf5'
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


## Compute PSNR and SSIM en test set
  
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
file = open(os.path.join(saveDir,'Illusions_deblurr.txt'),'w')
file.write('Mean PSNR'+str(np.mean(performance[:,0]))+'-'+str(np.std(performance[:,0]))) 
file.write('Mean SSIM'+str(np.mean(performance[:,1]))+'-'+str(np.std(performance[:,1]))) 
file.write(chkpt)
file.close() 

#np.save(saveDir+'jean_denoise_results',out) 
print('Saving Results ...')
np.save(os.path.join(saveDir,'Illusions_deblurr'),outIllusions) 
#np.save(saveDir+'jean_denoise_inputs',x_test) 
#np.save(saveDir+'jean_denoise_labels',y_test) 

#print('Loss function:'+str(score)+' Mean PSNR:'+str(meanPsnr)+' std:'+str(stdPsnr))                    