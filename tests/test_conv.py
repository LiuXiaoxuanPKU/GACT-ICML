import numpy as np
import torch
import torch.nn.functional as F
import pickle as p

data = p.load(open('data/cifar100/train','rb'),encoding='latin1')
im = np.float32(data['data'])/255.
im = im - np.mean(im,1,keepdims=True)
print(im.shape)
print(np.mean(im,1,keepdims=True))
train_im = np.reshape(im,[-1,3,32,32])
train_im = np.transpose(train_im,[0,2,3,1]).copy()
train_lb = np.reshape(np.int32(data['fine_labels']),[-1,1,1,1])

data = p.load(open('data/cifar100/test','rb'),encoding='latin1')
im = np.float32(data['data'])/255.
im = im - np.mean(im,1,keepdims=True)
val_im = np.reshape(im,[-1,3,32,32])
val_im = np.transpose(val_im,[0,2,3,1]).copy()
val_lb = np.reshape(np.int32(data['fine_labels']),[-1,1,1,1])

kernel = np.load('checkpoints.npz')['0']
kernel = torch.Tensor(kernel)
data = torch.Tensor(val_im[:1])

kernel = kernel.permute([3, 2, 0, 1])
data = data.permute([0, 3, 1, 2])

output = F.conv2d(data, kernel, padding=1)

