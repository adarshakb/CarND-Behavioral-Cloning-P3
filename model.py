"""
IMPORTS AND CONSTANTS/HELPERS
------------------------------------------------------------------------------------------------------------------------
"""
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import os.path
import cv2
import numpy as np
import sklearn
from keras.callbacks import ModelCheckpoint
from random import randint
import json
import os
import csv
from keras.models import Sequential
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Lambda, ELU
from keras.layers.normalization import BatchNormalization
from random import shuffle
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
#sys.path.append('/Users/abadarinath/Applications/anaconda/envs/UdacityNanoCar/lib/python3.5/site-packages')

DATA_DIR = ['../CarND-Behavioral-Cloning-Data/data_given/','../CarND-Behavioral-Cloning-Data/trained_data/leftTrack/','../CarND-Behavioral-Cloning-Data/trained_data/rightTrack/']
FLIPPED_IMG_APPEND = "IMG/FLIP/"
SHIFT_IMG_APPEND = "IMG/SHIFT/"
ch, row, col = 3, 32, 32  # image format

def getDrivingLog(i):
    return DATA_DIR[i] + 'driving_log.csv'
def getFinalDrivingLog(i):
    return DATA_DIR[i] + 'final_driving_log.csv'
def getImgDir(i):
    return DATA_DIR[i] + 'IMG/'
"""
IMPORTS AND CONSTANTS/HELPERS
------------------------------------------------------------------------------------------------------------------------
"""

"""
Function to crop image and convert to a 32*32*3 image
"""
def preprocessInputImage(src):
    row,col = 32,32
    img = cv2.imread(os.path.abspath(src))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    crop_img = img[50:125, 0:320]
    img = cv2.resize(crop_img, (row,col))
    return img

"""
A Simpe model with lamda
"""
def getModel():
	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row,col,ch)))
	model.add(Flatten())
	model.add(Dense(512, activation='elu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))

	return model
"""
The generator function to return Data in batches
"""
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[1]
                center_image = preprocessInputImage(name) # PREPROCESS DATA
                try:
                    center_angle = float(batch_sample[3])
                except ValueError:
                    center_angle = 0
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)



"""
Main function to execute
"""
def main():
	### LOAD ALL THE FINAL CSV FILES ###
	samples = []
	for i in range(len(DATA_DIR)):
	    print("reading",getFinalDrivingLog(i))
	    with open(getFinalDrivingLog(i)) as csvfile:
	        reader = csv.reader(csvfile)
	        for line in reader:
	            if line[1] == 'imgSrc':
	                pass
	            else:
	                line[1] = DATA_DIR[i] + line[1]
	                samples.append(line)
	    print("read - ",len(samples))

	print("Total samples", len(samples))
	

	### USE GENERATOR TO SPLIT TRAINING AND VALIDATION SETS ###
	train_samples, validation_samples = train_test_split(samples, test_size=0.1)
	train_generator = generator(train_samples, batch_size=256)
	validation_generator = generator(validation_samples, batch_size=256)

	## COMPILE AND FIT THE MODEL ###
	model = getModel()
	model.compile(optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='mse')
	model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10,verbose = 1)


	### SAVE THE MODEL ###
	if not os.path.exists("./output"): 
	    os.makedirs("./output")
	with open('model.json', 'w') as f:
	    json.dump(model.to_json(), f)
	# Save model weights to file
	model.save('model.h5')

	print("Saved model to disk")


if __name__ == "__main__":
    main()
