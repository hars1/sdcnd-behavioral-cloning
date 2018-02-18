# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/paper.png "Model Visualization"
[image2]: ./images/hist.png "Hist Samples"
[image3]: ./images/image.jpg "Image"
[image4]: ./images/flip_image.jpg "Flip Image"
[image5]: ./images/center.jpg "Recovery Image"
[image6]: ./images/center-cropped.jpg "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3  filter sizes


| Layer (type)               |  Output Shape          |    Param #   |
|----------------------------|------------------------|--------------|
| cropping2d_1 (Cropping2D)  |  (None, 65, 320, 3)    |    0		 |
| lambda_1 (Lambda)          |  (None, 65, 320, 3)    |    0         |
| conv2d_1 (Conv2D)          |  (None, 31, 158, 24)   |    1824      |
| conv2d_2 (Conv2D)          |  (None, 14, 77, 36)    |    21636     |
| conv2d_3 (Conv2D)          |  (None, 5, 37, 48)     |    43248     |
| conv2d_4 (Conv2D)          |  (None, 3, 35, 64)     |    27712     |
| conv2d_5 (Conv2D)          |  (None, 1, 33, 64)     |    36928     |
| flatten_1 (Flatten)        |  (None, 2112)          |    0         |
| dropout_1 (Dropout)        |  (None, 2112)          |    0         |
| dense_1 (Dense)            |  (None, 100)           |    211300    |
| dense_2 (Dense)            |  (None, 50)            |    5050      |
| dropout_2 (Dropout)        |  (None, 50)            |    0         |
| dense_3 (Dense)            |  (None, 10)            |    510       |
| dense_4 (Dense)            |  (None, 1)             |    11        |




Total params: 348,219

Trainable params: 348,219

Non-trainable params: 0
#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 79 and 82).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 128-129). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 134).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The final solution was started by a simple model containing a single convolution layer, and gradually introducing more complexity. As suggested by Udacity, the NVidia model (https://arxiv.org/pdf/1604.07316v1.pdf) was implemented with a slight addition i.e. L2 Regularization and dropout layers. The decision to introduce the L2 Regularization was in order to avoid overfitting. In addition, an image-cropping layer and a normalization of the date at the beginning of the network was introduced. Last but not least, a new layer at the end was added in order to have a single output as it was required i.e. the steering angle.


![alt text][image1]

                     NVIDIA CNN from the paper


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24)

The data is normalized in the model using a Keras lambda layer (code line 72).

| Layer (type)               |  Output Shape          |    Param #   |
|----------------------------|------------------------|--------------|
| cropping2d_1 (Cropping2D)  |  (None, 65, 320, 3)    |    0		 |
| lambda_1 (Lambda)          |  (None, 65, 320, 3)    |    0         |
| conv2d_1 (Conv2D)          |  (None, 31, 158, 24)   |    1824      |
| conv2d_2 (Conv2D)          |  (None, 14, 77, 36)    |    21636     |
| conv2d_3 (Conv2D)          |  (None, 5, 37, 48)     |    43248     |
| conv2d_4 (Conv2D)          |  (None, 3, 35, 64)     |    27712     |
| conv2d_5 (Conv2D)          |  (None, 1, 33, 64)     |    36928     |
| flatten_1 (Flatten)        |  (None, 2112)          |    0         |
| dropout_1 (Dropout)        |  (None, 2112)          |    0         |
| dense_1 (Dense)            |  (None, 100)           |    211300    |
| dense_2 (Dense)            |  (None, 50)            |    5050      |
| dropout_2 (Dropout)        |  (None, 50)            |    0         |
| dense_3 (Dense)            |  (None, 10)            |    510       |
| dense_4 (Dense)            |  (None, 1)             |    11        |



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, two laps were recorded in the Udacity simulator: one in a clock-wise direction and other counter-clockwise.The goal was to avoid the model to the biased towards left turns (clock-wise) or right turns (counter clock-wise).
Each data sample contains the steering angle/measurement as well as three images captured from three cameras installed at three different locations in the car [left, center, right]. To augment the data set, a flipping of the images and a change the sign of the steering angle was performed (see model.py lines 105-106). A histogram of the steering angle/measurements data is shown below.

![alt text][image2]

A python generator was used to generate samples for each batch of data that would be fed when training and validating the network. A generator is usefull in order not to store a lot of data
unnecessarily and only use the memory that we need to use at a time.

The data was randomly shuffled before  splitting it into training data (80 %) and validation data (20%) (see in model.py line 121).




To augment the data sat, I also flipped images and angles thinking that this would add more data and help the network generalize better. ... For example, here is an image that has then been flipped:

![alt text][image3]
![alt text][image4]

The steps above are part of the model itself and with that applied on the training, the validation set and also available while driving in autonomous mode using the model.

Original image:

![alt text][image5]

Image cropped:

![alt text][image6]



The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 and the MSE loss for training and validation decreased during the epochs.



#### Possible Improvements


* collect or generate more data by running on different track
* Use a pretrained network to start with (Transfer learning)