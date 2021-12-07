# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 20:20:16 2021

@author: Manuel
"""

# Libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# Hyperparameters
batchSize = 64
imageSize = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Input images pre-processing
transform = transforms.Compose(
    [
     transforms.Resize(imageSize), 
     transforms.ToTensor(), 
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     ]
)

# Loading the dataset, applying some pre-processing
dataset = dset.CIFAR10(
    root = './data', 
    download = True, 
    transform = transform
)
# will load the data batch by batch
dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size = batchSize, 
    shuffle = True, 
    num_workers = 2
) 

#function to initialize weights using pytorch's methods
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator( nn.Module ) :
    def __init__(self):
        super( Generator, self ).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels = 100, 
                out_channels = 512, 
                kernel_size= 4,
                stride = 1,
                padding = 0,
                bias = False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d( 512, 256, 4, 2, 1, bias = False ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d( 256, 128, 4, 2, 1, bias = False ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d( 128,  64, 4, 2, 1, bias = False ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(  64,   3, 4, 2, 1, bias = False ),
            nn.Tanh()
        )
    def forward(self, input):
        output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super( Discriminator, self ).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels = 3, 
                out_channels = 64, 
                kernel_size= 4,
                stride = 2,
                padding = 1,
                bias = False
            ),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d( 64, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d( 128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d( 256, 512, 4, 2, 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d( 512, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)

# Creating a generator
net_generator = Generator().to(device)
net_generator.apply(weights_init)

# Creating a discriminator
net_discriminator = Discriminator().to(device)
net_discriminator.apply(weights_init)

#Training parameters
criterion = nn.BCELoss()

discriminator_optimizer = optim.Adam(
    net_discriminator.parameters(),
    lr = 0.0002,
    betas = (0.5,0.999)
)
generator_optimizer = optim.Adam(
    net_generator.parameters(),
    lr = 0.0002,
    betas = (0.5,0.999)
)

epochs = 25


for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0): 
        # data is a minibatch, i is the index of the loop
        # 1st Step: Updating the weights of the neural network of the discriminator
        net_discriminator.zero_grad()
        
        #train with real images
        real_image, _ = data # _ would be labels
        input_image = Variable(real_image)
        target = Variable(torch.ones(input_image.size()[0]) )         
        output = net_discriminator(input_image.to(device))
        discriminator_error_real = criterion(output.to(device), target.to(device))
        #train with fake images
        noise = Variable(torch.randn(input_image.size()[0], 100, 1, 1) )    
        fake_image = net_generator(noise.to(device))
        target = Variable(torch.zeros(input_image.size()[0]) )
        output = net_discriminator(fake_image.detach().to(device))
        discriminator_error_fake = criterion(output.to(device), target.to(device))
        
        # total error backprop
        discriminator_error = discriminator_error_real+discriminator_error_fake
        discriminator_error.backward()
        discriminator_optimizer.step()
        
        # 2nd Step: Updating the weights of the neural network of the generator
        net_generator.zero_grad()
        target = Variable(torch.ones(input_image.size()[0]) )         
        output = net_discriminator(fake_image.to(device))
        generator_error = criterion(output.to(device), target.to(device))
        generator_error.backward()
        generator_optimizer.step()
        
        # 3rd Step: Printing the losses and saving the real images and the generated images of the minibatch every 100 steps
        print( '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (
                epoch, 
                epochs, 
                i, 
                len(dataloader), 
                discriminator_error.data, 
                generator_error.data
            )
        )
        if i % 100 == 0:
            vutils.save_image(
              real_image, 
              '%s/real_samples.png' % './results',
              normalize = True
            )
            fake = net_generator(noise.to(device))
            vutils.save_image(
              fake.data, 
              '%s/fake_samples_epoch_%03d.png' % ('./results', epoch),
              normalize = True
            )
            torch.save(net_discriminator.state_dict(), './results/discriminator.pth')
            torch.save(discriminator_optimizer.state_dict(), './results/discriminator_optim.pth')
            torch.save(net_generator.state_dict(), './results/generator.pth')
            torch.save(generator_optimizer.state_dict(), './results/generator_optim.pth')
        