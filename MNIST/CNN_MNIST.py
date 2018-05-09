#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: jie
"""
import torch, numpy as np, os
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


# Hyper Parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data/',
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data/',
                           train=False, 
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)



####################
#### CNN models ####
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Transforms a 1-channel image to 16 channels, using 5x5 kernels
        # Output is equivalent to a 16 channel image, of half the size (due to max pooling)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            # Batchnorm approximately normalizes input to zero-mean & unit-variance
            # Keeps running values for mean and variance
            # Also linearly transforms the output with learnable params
            nn.BatchNorm2d(16), ### !!!!!
            nn.ReLU(),
            nn.MaxPool2d(2))   # Max pooling (shrinks output by 1/2, to 14x14)
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)) # Shrink to 7x7
        
        # Input: equivalent of a 7x7 image with 32 channels
        # Output: 10 units for 10 digit classes
        self.fc = nn.Linear(7*7*32, 10)
    
    # Define the forward pass
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
cnn = CNN()



#######################
# Load trained model###
modelFilename = 'mnist-cnn.model'

if os.path.exists(modelFilename):
    cnn.load_state_dict(torch.load(modelFilename))

# Or train from scratch
else:   
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
        
    ####################
    # Train the Model #
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)
            
            # Forward + Backward + Optimize
            optimizer.zero_grad() # Clear stored gradients
            outputs = cnn(images) # CNN forward pass
            loss = criterion(outputs, labels) # Calculate error
            loss.backward() # Compute gradients
            optimizer.step() # Update weights
            
            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                       %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
    


##########################
# Save the Trained Model #
if not os.path.exists(modelFilename):
    torch.save(cnn.state_dict(), modelFilename)
    


####################
### Test the Model ###
# Change model to 'eval' mode
# This affects the batch normalization layer, so that it uses the mean/variance 
# obtained in training
cnn.eval()  
correct = 0
total = 0

for images, labels in test_loader:
    images = Variable(images)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))


########################################################################
###PART 2. visualize a training image through the convolution layers ###

import matplotlib.pyplot as plt
trainIter = iter(train_loader)  # Read in a training image 
images, labels = trainIter.next() 
img = torch.unsqueeze(Variable(images[0]), 0)


fig, ax = plt.subplots()
ax.imshow(img.data.numpy().reshape((28,28)), cmap = plt.cm.gray)
print('Input image')
print(img.shape) # 1 image, 1 color channel, WxH = 28x28
plt.show()


img_after_layer1 = cnn.layer1(img)
print('Image after layer 1')
print(img_after_layer1.shape) # 1 image, 16 channels, WxH = 14x14
nChannels = img_after_layer1.data.shape[1]
f, axarr = plt.subplots(4, 4, figsize=(8,8))
for i in range(16):
    axarr[i//4,i%4].imshow(img_after_layer1.data[0,i,:,:].numpy().reshape((14,14)),
                           cmap = plt.cm.gray, interpolation='nearest', aspect='equal')
for ax in axarr.flatten():
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()



print('Image after layer 2')
img_after_layer2 = cnn.layer2(img_after_layer1)
print(img_after_layer2.shape) # 1 image, 32 channel, WxH = 7x7
f2, axarr2 = plt.subplots(4, 8, figsize=(10,4))
for i in range(32):
    axarr2[i//8,i%8].imshow(img_after_layer2.data[0,i,:,:].numpy().reshape((7,7)),
                            cmap = plt.cm.gray, interpolation='nearest', aspect='equal')
for ax in axarr2.flatten():
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()





