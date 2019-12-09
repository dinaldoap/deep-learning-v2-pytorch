# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'intro-to-pytorch'))
	print(os.getcwd())
except:
	pass
#%%
from IPython import get_ipython

#%% [markdown]
# # Classifying Fashion-MNIST
# 
# Now it's your turn to build and train a neural network. You'll be using the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist), a drop-in replacement for the MNIST dataset. MNIST is actually quite trivial with neural networks where you can easily achieve better than 97% accuracy. Fashion-MNIST is a set of 28x28 greyscale images of clothes. It's more complex than MNIST, so it's a better representation of the actual performance of your network, and a better representation of datasets you'll use in the real world.
# 
# <img src='assets/fashion-mnist-sprite.png' width=500px>
# 
# In this notebook, you'll build your own neural network. For the most part, you could just copy and paste the code from Part 3, but you wouldn't be learning. It's important for you to write the code yourself and get it to work. Feel free to consult the previous notebooks though as you work through this.
# 
# First off, let's load the dataset through torchvision.

#%%
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
import helper

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

#%% [markdown]
# Here we can see one of the images.

#%%
image, label = next(iter(trainloader))
helper.imshow(image[0,:]);

#%% [markdown]
# ## Building the network
# 
# Here you should define your network. As with MNIST, each image is 28x28 which is a total of 784 pixels, and there are 10 classes. You should include at least one hidden layer. We suggest you use ReLU activations for the layers and to return the logits or log-softmax from the forward pass. It's up to you how many layers you add and the size of those layers.

#%%
class Classifier(nn.Module):
  def __init__(self):
    super().__init__()
    input_size = image[0].shape[1] * image[0].shape[2]
    hidden_layer_size = 128
    output_layer_size = 10
    self.model = nn.Sequential(nn.Linear(input_size, hidden_layer_size),
                      nn.ReLU(),
                      nn.Linear(hidden_layer_size, output_layer_size),
                      nn.ReLU(),
                      nn.LogSoftmax(dim=1))

  def forward(self, x):
    x = x.view(x.shape[0], -1)
    return self.model.forward(x)

#%% [markdown]
# # Train the network
# 
# Now you should create your network and train it. First you'll want to define [the criterion](http://pytorch.org/docs/master/nn.html#loss-functions) ( something like `nn.CrossEntropyLoss`) and [the optimizer](http://pytorch.org/docs/master/optim.html) (typically `optim.SGD` or `optim.Adam`).
# 
# Then write the training code. Remember the training pass is a fairly straightforward process:
# 
# * Make a forward pass through the network to get the logits 
# * Use the logits to calculate the loss
# * Perform a backward pass through the network with `loss.backward()` to calculate the gradients
# * Take a step with the optimizer to update the weights
# 
# By adjusting the hyperparameters (hidden units, learning rate, etc), you should be able to get the training loss below 0.4.

#%%
# Create the network, define the criterion and optimizer
model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


#%%
# Train the network here
for epoch in range(10):
  running_loss = 0
  for images, labels in trainloader:
    outputs = model.forward(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
  else:
    print('Running loss in epoch {}: {}'.format(epoch, running_loss/len(trainloader)))

#%%
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import helper
import math

# Test out your network!

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
# Convert 2D image to 1D vector
img = img.resize_(1, 784)

# Calculate the class probabilities (softmax) for img
ps = torch.exp(model.forward(img))

# Plot the image and probabilities
helper.view_classify(img.resize_(1, 28, 28), ps, version='Fashion')




# %%
