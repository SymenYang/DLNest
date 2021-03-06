import torch
import torch.nn as nn
import torch.distributed as dist
from .ModelBase import ModelBase
from abc import ABCMeta, abstractmethod

class ModelBaseTorch(ModelBase):
    def __init__(self,_envType : str,rank = -1,worldSize = -1):
        super(ModelBaseTorch,self).__init__(_envType = _envType,rank = rank,worldSize = worldSize)
        self.__modelList = []
    
    def DDPOperation(self,rank : int):
        pass

    def register(self,model : nn.Module,syncBN : bool = False):
        if self._envType == "DDP":
            model = model.cuda()
            if syncBN:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.__modelList.append(model)
            try:
                model = nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[self._rank],
                    output_device=self._rank
                )
                return model
            except AssertionError:
                return model
        else:
            if self._envType != "CPU":
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

    def _reduceSum(self,tensor):
        """
        ALPHA VERSION FUNCTION

        reduce tensor if using DDP,
        if not using DDP, do nothing.
        """
        if self._envType == "DDP":
            dist.reduce(tensor,0)
        return tensor

    def _reduceMean(self,tensor):
        """
        ALPHA VERSION FUNCTION

        reduce tensor if using DDP,
        if not using DDP, do nothing.
        """
        if self._envType == "DDP":
            dist.reduce(tensor,0)
            if self._rank == 0:
                tensor = tensor / self._worldSize
        return tensor