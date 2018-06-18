
# coding: utf-8

# In[1]:

# Importing all the modules needed 
import csv
import cv2
import numpy as np
import os
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.layers.advanced_activations import ELU
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Cropping2D
from keras.layers import Activation, Dropout, MaxPooling2D, Lambda
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt 
from keras.callbacks import ModelCheckpoint, Callback


# Specifying the paths for images and driving log obtained from the simulator and udacity data
img_path = './IMG/'
csv_path = './driving_log.csv'


image_paths = []
angles = []

# Reading the data from the driving log
with open('./driving_log.csv', newline = '') as csvfile:
    driving_data = list(csv.reader(csvfile, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))
 
    for row in driving_data[1:]:
        if float(row[6]) < 0.1:                                # Ignore the data where velocity is less than 0.1 when car is not moving
            continue
        image_paths.append(img_path + row[0].split('/')[-1])
        angles.append(float(row[3]))
        image_paths.append(img_path + row[1].split('/')[-1])   
        angles.append(float(row[3]) + 0.25)                    # adding correction factor to steer angles in order to use left camera images
        image_paths.append(img_path + row[2].split('/')[-1])
        angles.append(float(row[3]) - 0.25)                    # adding correction factor to steer angles in order to use right camera images
    
image_paths = np.array(image_paths)                            # all the image paths for center, left and right images together 
angles = np.array(angles)
print( image_paths.shape, angles.shape)


# Looking at the distribution of steering angle data
num_bins = 17
avg_samples_per_bin = len(angles)/num_bins
hist, bins = np.histogram(angles, num_bins)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(angles), np.max(angles)))


def brightness_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # convert from RGB to HSV color
    brightness_random =  .23 + np.random.uniform() # random brightness 
    image[:,:,2] = image[:,:,2]*brightness_random  # add brightness to the image
    image[:,:,2][image[:,:,2]>255] = 255           # keep image at 255 if greater than that 
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB) # convert back to RGB from HSV
    return image

def shadow_augmentation(image):
    '''refer to: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
    '''
    height, width, _ = image.shape
    imgtop_y = width*np.random.uniform() # 320
    imgtop_x = 0
    imgbot_y = width*np.random.uniform()
    imgbot_x = height # 160
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image[:,:,1]
    mask_X = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    mask_Y = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
    shadow_mask[((mask_X - imgtop_x)*(imgbot_y - imgtop_y) - (imgbot_x - imgtop_x)*(mask_Y - imgtop_y) >=0)] = 1
    if np.random.randint(2) == 1:
        condition_1 = shadow_mask == 1
        condition_0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            image[:,:,1][condition_1] = image[:,:,1][condition_1]*.5 # random brightness
        else:
            image[:,:,1][condition_0] = image[:,:,1][condition_0]*.5 
    image = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)
    return image 


def process_images(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)		# convert BGR to RGB
    image = brightness_process(image)					# brightness process
    image = shadow_augmentation(image)                  # adding random shadow mask on the image
 
    return image 





# Function to use for the Keras Generator
def generate_data( image_paths, angles, batch_size = 128):
    image_paths, angles = shuffle(image_paths, angles)
    X,y = ([],[])                            
 
    while True:       
        for i in range(len(angles)):
            img = cv2.imread(image_paths[i])
            angle = angles[i]
            img = process_images(img)
            
            X.append(img)
            y.append(angle)
 
 
            if len(X) == batch_size:
                yield (np.array(X), np.array(y))
                X, y = ([],[])
                image_paths, angles = shuffle(image_paths, angles)
                  
                 
            # flip horizontally and invert steer angle, if magnitude is > 0.20
            if abs(angle) > 0.25:
                img = cv2.flip(img, 1)
                angle *= -1
                X.append(img)
                y.append(angle)
                if len(X) == batch_size:
                    yield (np.array(X), np.array(y))
                    X, y = ([],[])
                    image_paths, angles = shuffle(image_paths, angles)           

# Covnet Archtiteture, using the nVidia model with image sizes as obtained from simulator
def nVidiaModel():
    """
    Creates nVidia model
    """
    model = Sequential()                            
    model.add(Lambda(lambda x: (x/255.0)- 0.5, input_shape = (160, 320, 3)))                             
    model.add(Cropping2D(cropping = ((50,20), (0,0)))) 
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))                           
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))          
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))          
    model.add(Convolution2D(64,3,3, activation='relu'))           
    model.add(Flatten())
    model.add(Dense(100))          
    model.add(Dense(50))          
    model.add(Dense(10))
    model.add(Dense(1))
    return model



# Splitting data into train and validation set
image_paths_train, image_paths_val, angles_train, angles_val = train_test_split(image_paths, angles, test_size=0.2, random_state=23)
print(image_paths_train.shape, angles_train.shape, image_paths_val.shape, angles_val.shape)
print(len(image_paths_train))

                        
# Model Initialization
model = nVidiaModel()

# Compiling and training the model
model.compile(loss='mse', optimizer='adam')     

train_gen = generate_data(image_paths_train, angles_train, batch_size=64)
val_gen = generate_data(image_paths_val, angles_val, batch_size=64)

batch_size = 64
samples_per_epoch = (len(image_paths_train)//batch_size)*batch_size
nb_val_samples = len(image_paths_val)
 


# In[12]:


# Using the genertor to train the network
history = model.fit_generator(train_gen, validation_data=val_gen, nb_val_samples=nb_val_samples, samples_per_epoch=samples_per_epoch, 
                               nb_epoch=20, verbose=1)

print(model.summary())  

# Save model data
json_string = model.to_json()
with open('./modelcopy3.json', 'w') as f:
 f.write(json_string)

model.save('modelcopy3.h5')

# plot training and validation loses
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

