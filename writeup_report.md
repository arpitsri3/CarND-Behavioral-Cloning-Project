# **Behavioral Cloning** 

## Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./cnn-architecture-624x890.png "Model Visualization"
[image2]: ./center_2016_12_01_13_37_57_790.jpg "Training Data Visualization"
[image3]: ./center_2016_12_01_13_37_57_790_cropped.jpg "Training Data Cropped"

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
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing - python drive.py model.h5 .

#### 3. Submission code is usable and readable

The model.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

For the purpose of the project I have used the architecture published by the Team at Nvidia which was also suggested by David in the classroom.

Here's the architecture:

![alt text][image1]

The changes that I have done are as follows:

1. I have added a lambda layer to pre process the incoming data as suggested during the classroom.
2. I have added a Cropping 2D layer to crop the images in exactly the same way as suggested during the classroom to help the network focus    on the road ahead.
3. I have added a L2 Regularizer to the Convolutional and Fully connected layers as my network was initially overfitting the test data        (more on that in the next section).
4. I have kept the number of epochs as 5.
5. The batch size has been kept as 33.
6. The optimizer that has been used is the 'Adam' optimizer with the default learning rate (I tried a smaller learning rate but i didn't      feel that the results of that were very different for this case)
7. I have used 'Relu' functions for activation instead of 'ELU' as i saw better results with 'RELU' activation.

Throught the period of working on the project I tried out a number of different variations of the architecture . These architectures are saved in the folder named 'Models' on the github repo. The architecture i finally used is the main 'model.py' one.

#### 2. Attempts to reduce overfitting in the model

To combat overfitting the main tool that has been impemented is the L2 regularizer. The L2 regularization technique which has been suggested in the classroom right from the project 2 turned out to be really decent in tackling overfitting on a centre-biased dataset. I tried dropout and max-pooling layers number of times while playing with the project architecture but generally found that network got confused at sharp turns or in a tricky landscape with this setup.

#### 3. Model parameter tuning

The model used an adam optimizer, and the default learning rate was used .

#### 4. Appropriate training data

The training data that I have used for the purpose of this project is the one provided by Udacity. This is because of two reasons:
1. I wasn't very happy with the way I was driving while trying to record data. I am not good at running cars in simulations and tend to        loose control of the simulated car often so I was concerned that the data used would set a bad example for the network.
2. A possible reason for why I was driving badly could be that my machine tends to lag a lot and was not equipped well enough to handle the    simulator very well, I feel.

Thus, I decided to use the data provided by Udacity and split it into Training and Validation data with a factor of 0.21

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to tweak an existing architecture to obtain good results.

My first step was to use a convolution neural network model similar to the Nvidia architecture. I thought this model might be appropriate because it had been explicitly mentioned in the classroom and was sufficiently powerful for our problem.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I used the images from both cemtre, left and right cameras.

There original model had a fair bit of overfitting happening with the model getting confused at sharp turns and going off track once. Again , as mentioned above i tried a number of tweaks (which may be viwed in the 'Models' folder) and ultimately settled for using the L2 regularizer.

The final step was to run the simulator to see how well the car was driving around track one. I really enjoyed tweaking the model and the hyperparameters to see in realtime what kind of effect the process had on the car. 

I have used data augmentation to augment the dataset to 6 times the original size and I am really happy to see that more data means better results.  I'll discuss this in more detail in the upcoming section on Training Data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. I am specially happy to see the sharp turns the vehicle takes some times to avoid going off the track at sharp turns.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes :

![alt text][image1]

Additionally, as discussed above the architecture has L2 regularizations enabled on the CN and Fully connected layers.

#### 3. Creation of the Training Set & Training Process

Due to the reasons and constraints mentioned above I have used the dataset provided by Udacity.

I have done data Augmentation on the above as follows, to drastically change the size of data by a factor of 6 :

1. In the generator function I have used the images from all three cameras.
2. I am adding a correction factor of 0.25 - as an assumption - to the steering angle for the images from left and right cameras.
3. Thus, for a batch_size of 33 , the batch gets enlarged by 33*3.
4. After this , for each image in this dataset of 99 images, i flip the image (as done by David in the classroom) and add a corresponding steering angle multiplied by -1. 
5. Thus , a batch of 33 finally contains 99*2 datapoints after the above process. There's an increase to 33x6= 198 Datapoints.

After the augmentation process, I had 38088 number of data points. I then preprocessed this data by by doing Lambda normalization and Cropping, which have again been explicitly discussed in the classroom. I'd like to say that this data augmentation , more than anything else, helped the most in training the network effectively.

Here's a visualization of the original and cropped image at a sharp turn:

Original Image:
![alt text][image2]

Cropped Image:
![alt text][image3]

The data has been shuffled and 21% of data has been split to validation data set. 

I used the augmented training data for training the model. The validation set helped determine if the model was over or under fitting. The final number of epochs that i went with was 5 as evidenced by hit and trial. I used an adam optimizer so that manually training the learning rate wasn't necessary.
