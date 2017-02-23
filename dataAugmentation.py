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

#sys.path.append('/Users/abadarinath/Applications/anaconda/envs/UdacityNanoCar/lib/python3.5/site-packages')

DATA_DIR = ['../CarND-Behavioral-Cloning-Data/data_given/','../CarND-Behavioral-Cloning-Data/trained_data/leftTrack/','../CarND-Behavioral-Cloning-Data/trained_data/rightTrack/']
FLIPPED_IMG_APPEND = "IMG/FLIP/"
SHIFT_IMG_APPEND = "IMG/SHIFT/"

def getDrivingLog(i):
    return DATA_DIR[i] + 'driving_log.csv'
def getFinalDrivingLog(i):
    return DATA_DIR[i] + 'final_driving_log.csv'
def getImgDir(i):
    return DATA_DIR[i] + 'IMG/'

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
