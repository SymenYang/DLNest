import torch
import torch.nn as nn
import torch.distributed as dist
from .RunnerBase import RunnerBase
from abc import ABCMeta, abstractmethod

class ModuleWrapper:
    """
    Contain at most two nn.modules, one is noraml torch module and the other one is the DDP shell contain this module
    Any attribute accessed by . are searched in the normal module and the __call__ function will call the DDP shell.
    Using singleCard to call the module's forward function.
    """
    keywords = ["_ModuleWrapper__M","_ModuleWrapper__DDP", "__dict__","singleCard"]

    def __init__(self, module : nn.Module, DDP_wrapper):
        self.__M = module
        self.__DDP = DDP_wrapper
    
    def __getattr__(self, name):
        if name in ModuleWrapper.keywords:
            return super().__getattr__(name)
        return self.__M.__getattr__(name)
    
    def __setattr__(self, name, value):
        if name in ModuleWrapper.keywords:
            self.__dict__[name] = value
            return
        return self.__M.__setattr__(name, value)

    def __delattr__(self, name):
        return self.__M.__delattr__(name)
    
    def __getattribute__(self, name):
        if name in ModuleWrapper.keywords:
            return super().__getattribute__(name)
        return self.__M.__getattribute__(name)
    
    def __call__(self, *args, **kwargs):
        if self.__DDP:
            return self.__DDP(*args, **kwargs)
        else:
            return self.__M(*args, **kwargs)
    
    def singleCard(self, *args, **kwargs):
        self.__M(*args, **kwargs)

class RunnerBaseTorch(RunnerBase):
    def __init__(self,args : dict, plugins = [], status = None):
        super(RunnerBaseTorch,self).__init__(args = args, plugins = plugins, status = status)
        self.__modelList = []
    
    def DDPOperation(self,rank : int):
        pass

    def register(self,model : nn.Module,syncBN : bool = False):
        if self._status.env == "DDP":
            self.__modelList.append(model)
            if isinstance(model, nn.Module):
                model = model.cuda()
                if syncBN:
                    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                try:
                    DDPmodel = nn.parallel.DistributedDataParallel(
                        model,
                        device_ids=[self._status.rank],
                        output_device=self._status.rank,
                        find_unused_parameters=True
                    )
                except AssertionError as e:
                    DDPmodel = None
            return ModuleWrapper(model, DDPmodel)
        else:
            self.__modelList.append(model)
            if isinstance(model, nn.Module):
                if self._status.env != "CPU":
                    model = model.cuda()
            return ModuleWrapper(model, None)

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
        if self._status.env == "DDP":
            dist.reduce(tensor,0)
        return tensor

    def _reduceMean(self,tensor):
        """
        ALPHA VERSION FUNCTION

        reduce tensor if using DDP,
        if not using DDP, do nothing.
        """
        if self._status.env == "DDP":
            dist.reduce(tensor,0)
            if self._status.rank == 0:
                tensor = tensor / self._status.worldSize
        return tensor
