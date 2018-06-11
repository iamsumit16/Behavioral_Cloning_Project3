
# coding: utf-8

# In[5]:


import csv
import cv2
import numpy as np


# In[4]:


lines = []
with open ('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


# In[6]:


images = []
measurements = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './IMG/'+ filename
    image = cv2.imread(current_path)
    images.append(image)
    
    measurement = float(line[3])
    measurements.append(measurement)
    
X_train = np.array(images)
y_train = np.array(measurements)





# In[12]:


print(X_train.shape, y_train.shape)


# In[10]:


from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D
from keras.layers import Activation, Dropout, MaxPooling2D
from keras.layers.normalization import BatchNormalization


# In[ ]:


print('Initialize Network Parameters')

batch_size = 100
epochs = 15
pool_size = (2, 2)
input_shape = X_train.shape[1:]

