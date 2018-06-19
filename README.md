# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Embedded Youtube Video for autonomous mode driving on the two tracks
---
[![Behavioral Cloning](http://img.youtube.com/vi/q7IJNIrmJ90/0.jpg)](https://www.youtube.com/watch?v=q7IJNIrmJ90 "Behavioral Cloning")

1. Introduction
---
This project asks us to apply the deep neural networks (convolutional network) to have a car drive itself autonomously using the Udacity simulator application. The simulator has training and autonomous modes. In the training mode, we can drive the vehicle on the given two tracks and record the data which consists of three set of images from three cameras mounted on left, center and right of the vehicle and a driving log .csv file which has the three images' paths, steering command, throttle command, brake and speed. For this project we will just be predicting steering command from the taken picture data and let the drive.py handle brake, throttle and speed/

We'll train the model developed using Keras and save the model information in a model.h5 file. 
 
Pipeline for this project consists of following steps:
* Collecting data from the simulator and using Udacity data
* Preprocessing the data
* Implementing a Generartor in Keras
* Define the model architecture
* Training and saving the model
* Driving the vehicle in autonomous mode and making a video of it
* Conculsion and discussion

Collecting data from the simulator and using Udacity data
---
Udacity has provided an image dataset and associated driving log file. Along with that I have also taken additional data by driving the vehicle on test tracks to produce the conditions of recovering back straight to the track when we are veered off the track and start approaching the curb. It allows model to learn the behavior on how to recover otherwise it might drive off the track or roll-over (in case of track two).

Preprocessing the data
---
In this step I read all the paths of images (center, left and right) in an array and all the corresponding steering angle commands in another. For using left and right camera, I have added a correction factor of 0.20(+.20 for left and -.20 for right image) which means that the car needs to recover to right if its veering to the left side and it needs to correct its position to left if going to right.

![alt text](https://github.com/iamsumit16/Udacity-CarND_Behavioral_Cloning_Project3/blob/master/data.png "Data distribution after adding correction factor to left and right images")
                                
I've included further processing like cropping the image to remove the unwanted part of the image like above the horizon or the hood part of the car in bottom of image and also, adding random brightness/darkness and shadow masks to the images to help model generalize better. This part of processing is a part of the generator function since I am not dealing with reading or storing the huge image data outside the generator to save the memory.

Implementing the Generator in Keras
---
Keras generators are a useful way to load the data one batch a time rather than loading it all at once. It takes in the arguments - image paths, angles and batch size. It reads the images using the image_paths array and angles and shuffles the data initially. 
Then I apply the image processing as mentioned above to the images and append them all together. Next, for the steering angle commands more than 0.20 I flip the images and reverse the angle sign and append them again. This helps in generating more data for angles other than straight (zero deg) and would also serve as data if the car had to drive in opposite to direction on the track for which the data was actually recorded. Each of the produced images and corresponding angles is added to a list and when the lengths of the lists reach batch_size the lists are converted to numpy arrays and yielded to the calling generator from the model. Finally, the lists are reset to allow another batch to be built and image_paths and angles are again shuffled. 
The generator feeds the model as it makes requests and then destroys it and goes over again until all the data is fed to the model.
Here's how the preprocessing of the data looks like before being fed and getting cropped in the model architecture.
![alt text](https://github.com/iamsumit16/Udacity-CarND_Behavioral_Cloning_Project3/blob/master/process1.png)
![alt text](https://github.com/iamsumit16/Udacity-CarND_Behavioral_Cloning_Project3/blob/master/process2.png)
![alt text](https://github.com/iamsumit16/Udacity-CarND_Behavioral_Cloning_Project3/blob/master/pro3.png)

Defining the model architecture
---
I have used here the [nVidia net](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) which was provided as one of the advanced architectures for end-to-end learning for self driving cars. However, the original model was trained over the images of size 66x200, I have used the image size as it was generated from the simulator.
NVidia Model:
![alt text](https://github.com/iamsumit16/Udacity-CarND_Behavioral_Cloning_Project3/blob/master/nvidiaNet.png)

Training, saving and running the model
---
The model was trained using model.fit() command in Keras for 20 epochs and it generated about 32000 test images to train over and about 8000 images to validate over. I used model.save command to save the model weight n .h5 format. The model training summary and losses are as below:

Epoch 1/20
31680/31680 [==============================] - 508s - loss: 0.0991 - val_loss: 0.0763
Epoch 2/20
31680/31680 [==============================] - 563s - loss: 0.0758 - val_loss: 0.0690
Epoch 3/20
31680/31680 [==============================] - 586s - loss: 0.0697 - val_loss: 0.0661
Epoch 4/20
31680/31680 [==============================] - 597s - loss: 0.0624 - val_loss: 0.0594
Epoch 5/20
31680/31680 [==============================] - 587s - loss: 0.0580 - val_loss: 0.0540
Epoch 6/20
31680/31680 [==============================] - 623s - loss: 0.0556 - val_loss: 0.0510
Epoch 7/20
31680/31680 [==============================] - 566s - loss: 0.0518 - val_loss: 0.0480
Epoch 8/20
31680/31680 [==============================] - 536s - loss: 0.0493 - val_loss: 0.0469
Epoch 9/20
31680/31680 [==============================] - 539s - loss: 0.0463 - val_loss: 0.0476
Epoch 10/20
31680/31680 [==============================] - 541s - loss: 0.0450 - val_loss: 0.0458
Epoch 11/20
31680/31680 [==============================] - 542s - loss: 0.0423 - val_loss: 0.0476
Epoch 12/20
31680/31680 [==============================] - 555s - loss: 0.0412 - val_loss: 0.0460
Epoch 13/20
31680/31680 [==============================] - 557s - loss: 0.0391 - val_loss: 0.0425
Epoch 14/20
31680/31680 [==============================] - 568s - loss: 0.0365 - val_loss: 0.0436
Epoch 15/20
31680/31680 [==============================] - 549s - loss: 0.0351 - val_loss: 0.0433
Epoch 16/20
31680/31680 [==============================] - 603s - loss: 0.0332 - val_loss: 0.0401
Epoch 17/20
31680/31680 [==============================] - 25307s - loss: 0.0309 - val_loss: 0.0423
Epoch 18/20
31680/31680 [==============================] - 18084s - loss: 0.0308 - val_loss: 0.0408
Epoch 19/20
31680/31680 [==============================] - 513s - loss: 0.0290 - val_loss: 0.0412
Epoch 20/20
31680/31680 [==============================] - 517s - loss: 0.0275 - val_loss: 0.0396
____________________________________________________________________________________________________
Layer (type) Output Shape Param # Connected to

====================================================================================================
lambda_1 (Lambda) (None, 160, 320, 3) 0 lambda_input_1[0][0]
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D) (None, 90, 320, 3) 0 lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D) (None, 43, 158, 24) 1824 cropping2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D) (None, 20, 77, 36) 21636 
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D) (None, 8, 37, 48) 43248 
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D) (None, 6, 35, 64) 27712 
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D) (None, 4, 33, 64) 36928 
____________________________________________________________________________________________________
flatten_1 (Flatten) (None, 8448) 0 convolution2d_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense) (None, 100) 844900 flatten_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense) (None, 50) 5050 dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense) (None, 10) 510 dense_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense) (None, 1) 11 dense_3[0][0]

====================================================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
____________________________________________________________________________________________________
None


![alt text](https://github.com/iamsumit16/Udacity-CarND_Behavioral_Cloning_Project3/blob/master/loss.png)


### Directions from Udacity on requirements, how to make the model and run it:
---

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

