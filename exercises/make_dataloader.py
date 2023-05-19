
###Préparation des datasets
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import os 
from torch import nn
import torch.utils.data


def make_dataloader(train_path,test_path,transformer,batch_size):


    trainset = datasets.ImageFolder(root=train_path,
                                    transform=transformer)
    
    testset = datasets.ImageFolder(root=test_path,
                                    transform=transformer)
    

    train_dl=DataLoader(dataset=trainset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=os.cpu_count())
    
    test_dl=DataLoader(dataset=testset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=os.cpu_count())
    
    
    return train_dl, test_dl, trainset.classes

