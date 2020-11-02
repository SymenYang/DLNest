import torch.utils.data as data
import torch
import numpy as np
import json

class DefaultDataset(data.Dataset):
    def __init__(self,args : dict):
        self.args = args
        self.length = 2
    
    def __getitem__(self,index : int):
        return np.array([0,1])
    
    def __len__(self):
        return self.length

class DatasetBase:
    def __init__(self,args : dict):
        """
        init dataset
        input: 
            args: dict of task argumentations
        
        output:
            dict,Dataloader,Dataloader
            dict: infomation which dataset want to tell model when model init
            Dataloaders: Dataloaders for training,validation. If no validation, return two same dataloaders
        """
        self.trainSet = DefaultDataset(args)
        self.valSet = DefaultDataset(args)
        self.trainLoader = torch.utils.data.DataLoader(
            self.trainSet
        )
        self.valLoader = torch.utils.data.DataLoader(
            self.valSet
        )
    
    def afterInit(self):
        return {},self.trainLoader,self.valLoader