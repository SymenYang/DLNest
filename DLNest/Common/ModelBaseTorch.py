import torch
import torch.nn as nn
from .ModelBase import ModelBase
from abc import ABCMeta, abstractmethod

class ModelBaseTorch(ModelBase):
    def __init__(self,envType : str,rank = -1):
        super(ModelBaseTorch,self).__init__(envType = envType,rank = rank)
        self.__modelList = []
    
    def DDPOperation(self,rank : int):
        pass

    def register(self,model : nn.Module,syncBN : bool = False):
        if self.envType == "DDP":
            model = model.cuda()
            if syncBN:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.__modelList.append(model)
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.rank],
                output_device=self.rank
            )
            return model
        else:
            if self.envType != "CPU":
                model = model.cuda()
            self.__modelList.append(model)
            return model

    def getSaveDict(self):
        stateDict = {}
        for i in range(len(self.__modelList)):
            stateDict[i] = self.__modelList[i].state_dict()
        return stateDict
    
    def loadSaveDict(self,saveDict):
        for i in range(len(self.__modelList)):
            self.__modelList[i].load_state_dict(saveDict[i])