from DLNest.Common.DatasetBase import DatasetBase

import torchvision
import torch
from torch.utils import data
from torchvision import datasets, transforms

class MNISTDataset(DatasetBase):
    def init(self,args : dict):
        transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.5],std=[0.5])])
        self.dataTrain = datasets.MNIST(root = args["dataset_config"]["data_root"],
                                  transform=transform,
                                  train = True,
                                  download = True)

        self.dataTest = datasets.MNIST(root = args["dataset_config"]["data_root"],
                                 transform = transform,
                                 train = False)

        trainSampler = self.getSampler(self.dataTrain)
        testSampler = self.getSampler(self.dataTest)
        if trainSampler is None: # DDP
            self.trainLoader = data.DataLoader(self.dataTrain,batch_size = args["dataset_config"]["batch_size"],shuffle = True)
            self.testLoader = data.DataLoader(self.dataTest,batch_size = args["dataset_config"]["batch_size"],shuffle = False)
        else:
            self.trainLoader = data.DataLoader(self.dataTrain,batch_size = args["dataset_config"]["batch_size"],sampler=trainSampler)
            self.testLoader = data.DataLoader(self.dataTest,batch_size = args["dataset_config"]["batch_size"],sampler=testSampler)

        return {},self.trainLoader,self.testLoader
