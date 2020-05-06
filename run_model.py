#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image  
import PIL


# In[2]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # number of hidden nodes in each layer (512)
        hidden_1 = 512
        hidden_2 = 512
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # linear layer (n_hidden -> 47)
        self.fc3 = nn.Linear(hidden_2, 47)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add output layer
        x = self.fc3(x)
        return x

model = Net()


# In[3]:


model.load_state_dict(torch.load('model.pt'))


# In[12]:


alphabet='abcdefghijklmnopqrstuvwxyz'
alpha_lower=list(alphabet)
alpha_upper=list(alphabet.upper())
num= list(range(0,10))
bad_lower = ['c','i','j','k','l','m','o','p','s','u','v','w','x','y','z']
valid_lower = [letter for letter in alpha_lower if letter not in bad_lower]
all_class = num+alpha_upper+valid_lower
indx_to_class = {num:all_class[num] for num in range(len(all_class))}
print(indx_to_class)


# In[14]:


model.eval()
image_path= 'test.png' #put image here
img = Image.open(image_path).convert('L')


# In[15]:


transform= transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor()])
data = transform(img).squeeze(0)


# In[16]:


output = model(data)
_, preds = torch.max(output,1)
indx= preds.item()
print(indx_to_class[indx])

