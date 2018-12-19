## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
import numpy as np


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
#       # output size = (W-F)/S +1 = (224-5)/1 +1 =220 
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        # after one pool layer, this becomes (32, 110, 110)
        
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.1)
        
        # output size = (W-F)/S +1 = (110-3)/1 +1 = 108
        # the output Tensor for one image, will have the dimensions: (64, 108, 108)
        # after one pool layer, this becomes (64, 54, 54)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.2)
        
        # output size = (W-F)/S +1 = (54-3)/1 +1 = 52
        # the output Tensor for one image, will have the dimensions: (128, 52, 52)
        # after one pool layer, this becomes (128, 26, 26)
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.3)
        
        # output size = (W-F)/S +1 = (26-1)/1 +1 = 26
        # the output Tensor for one image, will have the dimensions: (256, 26, 26)
        # after one pool layer, this becomes (256, 13, 13)
        
        self.conv4 = nn.Conv2d(128, 256, 1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout2d(0.4)
        
        # fully connected layers
        self.fc1 = nn.Linear(13*13*256, 1000)
        self.drop5 = nn.Dropout2d(0.5)
                
        self.fc2 = nn.Linear(1000, 1000)
        self.drop6 = nn.Dropout2d(0.6)
        
        # last layer with the maximum number of key points
        self.fc3 = nn.Linear(1000, 136)
        
        # Weight Initialization
        self.conv1.weight.data.uniform_(-0.03, 0.03)
        self.conv2.weight.data.uniform_(-0.03, 0.03)
        self.conv3.weight.data.uniform_(-0.03, 0.03)
        self.conv4.weight.data.uniform_(-0.03, 0.03)
        
        I.xavier_uniform_(self.fc1.weight, gain=np.sqrt(2))
        I.constant_(self.fc1.bias, 0.1)
        
        I.xavier_uniform_(self.fc2.weight, gain=np.sqrt(2))
        I.constant_(self.fc2.bias, 0.1)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool1(F.elu(self.conv1(x)))
        x = self.drop1(x)
        
        x = self.pool2(F.elu(self.conv2(x)))
        x = self.drop2(x)
        
        x = self.pool3(F.elu(self.conv3(x)))
        x = self.drop3(x)
        
        x = self.pool4(F.elu(self.conv4(x)))
        x = self.drop4(x)
        
        x = x.view(x.size(0), -1) # flatten
        x = F.relu(self.fc1(x))
        x = self.drop5(x)
        
        x = F.relu(self.fc2(x))
        x = self.drop6(x)
        
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
