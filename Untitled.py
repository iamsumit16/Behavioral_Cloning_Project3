
# coding: utf-8

# In[5]:


import csv
import cv2
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Cropping2D
from keras.layers import Activation, Dropout, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt 



def getMeasurements(data_path, skipHeader = False):
    
    lines = []
    with open(data_path + './driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        if skipHeader:
            next(reader, None)
        for line in reader:
            lines.append(line)
    return(lines)



def getImages(data_path):
    directory = [x[0] for x in os.walk(data_path)]
    data_directory = list(filter(lambda directory: os.path.isfile(directory + '/driving_log.csv'), directory)
    center_images =  []
    left_images = []
    right_images = []
    measurements_all = []
    
    for directory in data_directory:
        lines = getMeasurements(directory)
        center = []
        left = []
        right = []
        measurements = []  
         
        for line in lines:
            measurements.append(float(line[3])
            center.append(directory + '/' + line[0].strip())
            left.append(directory + '/' + line[1].strip()) 
            right.append(directory + '/' + line[2].strip()) 
        
         center_images.extend(center)
         left_images.extend(left)
         right_images.extend(right)
         measurements_all.extend(measurements)
    return(center_images, left_images, right_images, measurements_all)

  
 def combineImages(center, left, right, measurement, correction):
 """
 cobine the three camera images together using the correction factor
 """
     image_paths_all = []
     image_paths_all.extend(center)                          
     image_paths_all.extend(left)                           
     image_paths_all.extend(right)

     angles  = []
     angles.extend([ x + correction for x in measurement]) 
     angles.extend([ x - correction for x in measurement]) 
     return(image_paths_all, angles)                           

def preprocess_image():
    model = Sequential()                            
    model.add(Lambda(lambda x: (x/255.0)- 0.5, input_shape = (160, 320, 3)))                            
    model.add(Croppind2D(cropping = ((50,20), (0,0))))                            
    return(model)
                                
def generator( image_paths, angles, batch_size = 128):
    image_paths, angles = shuffle(image_paths, angles)
    X,y = ([],[])                            
                                
    while True:       
        for i in range(len(angles)):
            img = cv2.imread(image_paths[i])
            angle = angles[i]
            img = preprocess_image(img)
            
            X.append(img)
            y.append(angle)
                                
                                
            if len(X) == batch_size:
                yield (np.array(X), np.array(y))
                X, y = ([],[])
                image_paths, angles = shuffle(image_paths, angles)
                  
                                                
            # flip horizontally and invert steer angle, if magnitude is > 0.30
            if abs(angle) > 0.30:
                img = cv2.flip(img, 1)
                angle *= -1
                X.append(img)
                y.append(angle)
                if len(X) == batch_size:
                    yield (np.array(X), np.array(y))
                    X, y = ([],[])
                    image_paths, angles = shuffle(image_paths, angles)           
                               

print('Initialize Network Parameters')
def nVidiaModel():
    """
    Creates nVidia model
    """
    model = preprocess_image()
    model.add(Convolution2D(24,5,5, subsample=(2,2), border_mode='valid'))
    model.add(ELU())                            
    model.add(Convolution2D(36,5,5, subsample=(2,2), border_mode='valid'))
    model.add(ELU())           
    model.add(Convolution2D(48,5,5, subsample=(2,2), border_mode='valid'))
    model.add(ELU())
    model.add(Convolution2D(64,3,3))
    model.add(ELU())          
    model.add(Convolution2D(64,3,3))
    model.add(ELU())           
    model.add(Flatten())
    model.add(Dense(100))
    model.add(ELU())          
    model.add(Dense(50))
    model.add(ELU())          
    model.add(Dense(10))
    model.add(ELU()) 
    model.add(Dense(1))
    return model


# Reading images locations.
centerPaths, leftPaths, rightPaths, measurements = getImages('data')
imagePaths, measurements = combineImages(centerPaths, leftPaths, rightPaths, measurements, 0.2)
print('Total Images: {}'.format( len(imagePaths)))

# Splitting samples and creating generators.


train_samples, validation_samples = train_test_split(imagePaths, measurements, test_size=0.2)

print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))

train_generator = generator(train_samples, batch_size = 128)
validation_generator = generator(validation_samples, batch_size = 128)
                                
checkpoint = ModelCheckpoint('model{epoch:02d}.h5')
                                
# Model Initialization
model = nVidiaModel()

# Compiling and training the model
model.compile(loss='mse', optimizer='Nadam')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
                 len(train_samples), validation_data=validation_generator, \
                 nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

                                
print(model.summary())                                
model.save('model.h5')
model                                
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])

                                
                                
json_string = model.to_json()
    with open('./model.json', 'w') as f:
        f.write(json_string)       
                                
                                

