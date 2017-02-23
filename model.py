import sys
#sys.path.append('/Users/abadarinath/Applications/anaconda/envs/UdacityNanoCar/lib/python3.5/site-packages')

DATA_DIR = ['../CarND-Behavioral-Cloning-Data/data_given/','../CarND-Behavioral-Cloning-Data/trained_data/leftTrack/','../CarND-Behavioral-Cloning-Data/trained_data/rightTrack/']

def getDrivingLog(i):
    return DATA_DIR[i] + 'driving_log.csv'
def getFinalDrivingLog(i):
    return DATA_DIR[i] + 'final_driving_log.csv'
def getImgDir(i):
    return DATA_DIR[i] + 'IMG/'

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import randint
import numpy as np
import pandas as pd
# Visualizations will be shown in the notebook.
%matplotlib inline

#show given image img, or if img is None, load the img from src and show 
def showImage(imgName=None,img=None,title=None,i=0):
    if imgName != None:
        imgName=imgName.strip()
        img = mpimg.imread(DATA_DIR[i]+imgName)
    
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(title)    
    plt.show()

#load the CSV and output 5 random sample images
import os.path
import cv2
FLIPPED_IMG_APPEND = "IMG/FLIP/"
SHIFT_IMG_APPEND = "IMG/SHIFT/"
for i in range(len(DATA_DIR)):
    driving_log = pd.DataFrame.from_csv(getDrivingLog(i),index_col=False)
    n_driving_log = driving_log.shape[0]
    print(DATA_DIR[i])
    print("Given driving log details")
    print("Number of rows = ",n_driving_log)
    print("Number of features = ",driving_log.shape[1])
    print("A few random sample output")
    randomImgIndex = np.zeros(5)
    os.makedirs(os.path.dirname(DATA_DIR[i]+FLIPPED_IMG_APPEND), exist_ok=True)
    os.makedirs(os.path.dirname(DATA_DIR[i]+FLIPPED_IMG_APPEND+"/IMG/"), exist_ok=True)
    os.makedirs(os.path.dirname(DATA_DIR[i]+SHIFT_IMG_APPEND), exist_ok=True)


def fligImgAndSave(imgName,i):
    imgName=imgName.strip()
    #print(DATA_DIR[i]+FLIPPED_IMG_APPEND+imgName)
    if os.path.isfile(DATA_DIR[i]+FLIPPED_IMG_APPEND+imgName):
        return #skip if file already exist
    Image.open(DATA_DIR[i]+imgName).transpose(Image.FLIP_LEFT_RIGHT).save(DATA_DIR[i]+FLIPPED_IMG_APPEND+imgName)


def shiftImg(img,shift):
    rows,cols,ch = img.shape
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[50-shift,50],[200-shift,50],[50-shift,200]])
    M = cv2.getAffineTransform(pts1,pts2)
    return cv2.warpAffine(img,M,(cols,rows))
    
def shiftImgAndSave(imgName,shift,i):
    imgName=imgName.strip()
    os.makedirs(os.path.dirname(DATA_DIR[i]+SHIFT_IMG_APPEND+str(shift)+"/IMG/"), exist_ok=True)
    if os.path.isfile(DATA_DIR[i]+SHIFT_IMG_APPEND+imgName):
        return #skip if file already exist
    img = mpimg.imread(DATA_DIR[i]+imgName)
    dst = shiftImg(img,shift)
    mpimg.imsave(DATA_DIR[i]+SHIFT_IMG_APPEND+str(shift)+"/"+imgName,dst)
    #print(DATA_DIR[i]+SHIFT_IMG_APPEND+str(shift)+"/"+imgName)

for i in range(5):
    randomImgIndex[i] = randint(0,n_driving_log)
tmp=randomImgIndex[4]
driving_log = pd.DataFrame.from_csv(getDrivingLog(0),index_col=False)
showImage(driving_log['center'][tmp],None,driving_log['steering'][tmp])
from PIL import Image
trackImg = Image.open(DATA_DIR[0]+driving_log['center'][tmp]).transpose(Image.FLIP_LEFT_RIGHT).save(getImgDir(0)+'tmp_horizontal_flip.jpg')
showImage('IMG/tmp_horizontal_flip.jpg',None,"Flipped and saved imaged from above sample")


def addIntoFinalDrivingList(imgSrc_inp,steering_inp,throttle_inp,brake_inp,speed_inp):
    imgSrc.append(imgSrc_inp.strip())
    steering.append(steering_inp)
    throttle.append(throttle_inp)
    brake.append(brake_inp)
    speed.append(speed_inp)

def saveNewDrivingLog(i):
    df = pd.DataFrame(data={'imgSrc':imgSrc,'steering':steering,'throttle':throttle,'brake':brake,'speed':speed})
    df.to_csv(getFinalDrivingLog(i),index=False)

print("Converting and augmenting data")

for x in range(0,len(DATA_DIR)):
    #lets do the same for all the given images to augment the data
    imgSrc=list() # combine any image 
    steering=list()
    throttle=list()
    brake=list()
    speed=list()
    driving_log = pd.DataFrame.from_csv(getDrivingLog(x),index_col=False)
    n_driving_log = driving_log.shape[0]
    print(getDrivingLog(x),driving_log.columns)
    for i in range(n_driving_log):
        if i % 20 == 0:
            print(x,i,i/n_driving_log)
        # create adjusted steering measurements for the side camera images
        steering_center = driving_log['steering'][i]
        if (steering_center == 0 and np.random.choice([1,2,3,4,5,6,7,8,9,10]) >= 2) or driving_log['speed'][i] < 20 or driving_log['throttle'][i] < 0.25:# or steering_center > 0.6 or steering_center < -0.6:
            pass # ignore the 0 value input because we have too many of them and lets not overfit the data to 0
        else:
            correction = 0.25 # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            try:
                addIntoFinalDrivingList(driving_log['center'][i].strip(),steering_center,driving_log['throttle'][i],driving_log['brake'][i],driving_log['speed'][i])
                addIntoFinalDrivingList(driving_log['left'][i].strip(),steering_left,driving_log['throttle'][i],driving_log['brake'][i],driving_log['speed'][i])
                addIntoFinalDrivingList(driving_log['right'][i].strip(),steering_right,driving_log['throttle'][i],driving_log['brake'][i],driving_log['speed'][i])
            except AttributeError:
                print("ERROR OCCURED")
                continue
            fligImgAndSave(driving_log['center'][i],x)
            fligImgAndSave(driving_log['left'][i],x)
            fligImgAndSave(driving_log['right'][i],x)

            addIntoFinalDrivingList(FLIPPED_IMG_APPEND+driving_log['center'][i].strip(),-steering_center,driving_log['throttle'][i],driving_log['brake'][i],driving_log['speed'][i])
            addIntoFinalDrivingList(FLIPPED_IMG_APPEND+driving_log['left'][i].strip(),-steering_left,driving_log['throttle'][i],driving_log['brake'][i],driving_log['speed'][i])
            addIntoFinalDrivingList(FLIPPED_IMG_APPEND+driving_log['right'][i].strip(),-steering_right,driving_log['throttle'][i],driving_log['brake'][i],driving_log['speed'][i])


            #if steering_center > 0.11 or steering_center < -0.11:
            shift_correction = 0.004 # this is a parameter to tune
            shift = np.random.choice([-30,-20,-10,10,20,30]) # this is a parameter to tune

            shiftImgAndSave(driving_log['center'][i],shift,x)
            shiftImgAndSave(driving_log['left'][i],shift,x)
            shiftImgAndSave(driving_log['right'][i],shift,x)

            steering_center = driving_log['steering'][i] + (shift * shift_correction)
            steering_left = steering_center + correction + (shift * shift_correction)
            steering_right = steering_center - correction + (shift * shift_correction)

            addIntoFinalDrivingList(SHIFT_IMG_APPEND+str(shift)+"/"+driving_log['center'][i].strip(),steering_center,driving_log['throttle'][i],driving_log['brake'][i],driving_log['speed'][i])
            addIntoFinalDrivingList(SHIFT_IMG_APPEND+str(shift)+"/"+driving_log['left'][i].strip(),steering_left,driving_log['throttle'][i],driving_log['brake'][i],driving_log['speed'][i])
            addIntoFinalDrivingList(SHIFT_IMG_APPEND+str(shift)+"/"+driving_log['right'][i].strip(),steering_right,driving_log['throttle'][i],driving_log['brake'][i],driving_log['speed'][i])

    saveNewDrivingLog(x)
    print(getFinalDrivingLog(x) + " is saved with new augmented data for "+ DATA_DIR[x])


for i in range(len(DATA_DIR)):
    #load the CSV and output 5 random sample images
    print(getFinalDrivingLog(i))
    final_driving_log = pd.DataFrame.from_csv(getFinalDrivingLog(i),index_col=False)

    n_final_driving_log = final_driving_log.shape[0]
    print("Given FINAL driving log details")
    print("Number of rows = ",n_final_driving_log)
    print("Number of features = ",final_driving_log.shape[1])
    print("A few random sample output")

    plt.hist(final_driving_log['steering'],bins=100)
    plt.show()

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    # return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

#read a single image and dermine the clipping angle

tesImg = 'IMG/FLIP/IMG/right_2016_12_01_13_32_54_674.jpg'#final_driving_log['imgSrc'][0]
showImage(tesImg)

image = mpimg.imread(DATA_DIR[0]+tesImg)
image = grayscale(image)
region_of_intrest = [[0,50],[320,50],[320,125],[0,125]]
image = region_of_interest(image,np.array([region_of_intrest]))
plt.figure(1)
plt.imshow(image)#, cmap='gray')
plt.show()

# hence we can carry forward the "region of intrest" into the models forward


def preprocessInputImage(src):
    row,col = 32,32
    img = cv2.imread(os.path.abspath(src))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    crop_img = img[50:125, 0:320]
    img = cv2.resize(crop_img, (row,col))
#     plt.figure(1)
#     plt.imshow(img)#, cmap='gray')
#     plt.show()
    #print(np.asarray(img).shape)
    return img

image = Image.open(DATA_DIR[0]+tesImg).convert('RGB').crop((0, 50, 320, 125)).resize((32,32))
# print(img.shape)
plt.figure(1)
plt.imshow(image) #, cmap='gray')
plt.show()
image_array = np.asarray(image)#np.reshape(np.asarray(image),(row,col,1))
print(image_array.shape)

# print(DATA_DIR[0]+tesImg)
# preprocessInputImage(DATA_DIR[0]+tesImg)


import os
import csv
#Construct the keras model
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
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.1)

import cv2
import numpy as np
import sklearn

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
                #print(name)
                center_image = preprocessInputImage(name)
                try:
                    center_angle = float(batch_sample[3])
                except ValueError:
                    center_angle = 0
                images.append(center_image)
                angles.append(center_angle)
                #plt.figure(1)
                #plt.imshow(cv2.cvtColor(center_image,cv2.COLOR_BGR2RGB))
                #plt.show()
                #print('test',name,center_angle,center_image.shape)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=256)
validation_generator = generator(validation_samples, batch_size=256)

#print(next(train_generator))


ch, row, col = 3, 32, 32  # image format

# model = Sequential()
# # Preprocess incoming data, centered around zero with small standard deviation
# model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(row,col,ch),output_shape=(row,col,ch)))
# model.add(Convolution2D(4, 1, 1, subsample=(2, 2), border_mode='same',init = 'he_normal'))
# model.add(MaxPooling2D())
# model.add(Dropout(.5))
# model.add(ELU())
# # model.add(Convolution2D(32, 8, 8, subsample=(1, 1), border_mode='same',init = 'he_normal'))
# # model.add(BatchNormalization())
# # model.add(Dropout(.5))
# # model.add(ELU())
# # model.add(Convolution2D(64, 16, 16, subsample=(2, 2), border_mode="same",init = 'he_normal'))
# model.add(Flatten())
# model.add(Dropout(.5))
# model.add(ELU())
# model.add(Dense(256))
# model.add(Dropout(.5))
# # model.add(BatchNormalization())
# model.add(ELU())
# model.add(Dense(1))

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row,col,ch)))
# model.add(Convolution2D(3, 1, 1))
# model.add(Convolution2D(64, 7, 7, border_mode='valid', activation='elu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Convolution2D(128, 4, 4, border_mode='valid', activation='elu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Convolution2D(256, 4, 4, border_mode='valid', activation='elu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, activation='elu'))
model.add(Dropout(0.5))
# model.add(Dense(128, activation='elu'))
# model.add(Dropout(0.5))
# model.add(Dense(32, activation='elu'))
model.add(Dense(1))

# Recompile the model with a finer learning rate
model.compile(optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='mse')
print(model.summary())



history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10,verbose = 1)
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


from keras.callbacks import ModelCheckpoint
import json

if not os.path.exists("./output"): 
    os.makedirs("./output")

print("\nSaving model weights and configuration file.")

with open('model.json', 'w') as f:
    json.dump(model.to_json(), f)
# Save model weights to file
model.save('model.h5')

print("Saved model to disk")

