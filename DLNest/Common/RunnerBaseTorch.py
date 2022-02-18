import torch
import torch.nn as nn
import torch.distributed as dist
from .RunnerBase import RunnerBase
from abc import ABCMeta, abstractmethod
import typing
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from DLNest.Plugins.Utils.CheckPlugins import checkPlugins,checkDictOutputPlugins

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
    
    def __str__(self):
        if self.__DDP:
            return self.__DDP.__str__() + " in DLNest ModuleWrapper"
        else:
            return self.__M.__str__() + " in DLNest ModuleWrapper"

    def singleCard(self, *args, **kwargs):
        self.__M(*args, **kwargs)

class RunnerBaseTorch(RunnerBase):
    def __init__(self,args : dict, plugins = [], status = None):
        super(RunnerBaseTorch,self).__init__(args = args, plugins = plugins, status = status)
        self.__modelList = []
        self.__optimizerDict = {}
        self.__schedulerDict = {}
    
    def DDPOperation(self,rank : int):
        pass

    def __setattr__(self, name, value):
        if isinstance(value, nn.Module):
            super().__setattr__(name, self.register(value))
        elif isinstance(value, Optimizer):
            super().__setattr__(name, self.__registerOptimizer(value, name))
        elif isinstance(value, _LRScheduler):
            super().__setattr__(name, self.__registerLRScheduler(value, name))
        else:
            super().__setattr__(name,value)

    def register(self,module : nn.Module, syncBN : bool = False):
        assert isinstance(module, nn.Module), "Only nn.Module can be registered. You tried to register a " + module.__class__.__name__
        self.__modelList.append(module)
        if self._status.env == "DDP":
            if isinstance(module, nn.Module):
                model = module.cuda()
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
            if isinstance(module, nn.Module):
                if self._status.env != "CPU":
                    model = module.cuda()
            return ModuleWrapper(model, None)

    def __registerOptimizer(self, optimizer : Optimizer, name):
        self.__optimizerDict[name] = optimizer
        return optimizer

    def __registerLRScheduler(self, scheduler : _LRScheduler, name):
        self.__schedulerDict[name] = scheduler
        return scheduler

    def getSaveDict(self):
        stateDict = {}
        for i in range(len(self.__modelList)):
            stateDict[i] = self.__modelList[i].state_dict()

        stateDict["optimizer"] = {}
        for name in self.__optimizerDict:
            optimizer = self.__optimizerDict[name]
            try:
                stateDict["optimizer"][name] = optimizer.state_dict()
            except Exception as e:
                pass # may add some warning

        stateDict["scheduler"] = {}
        for name in self.__schedulerDict:
            scheduler = self.__schedulerDict[name]
            try:
                stateDict["scheduler"][name] = scheduler.state_dict()
            except Exception as e:
                pass # may add some warning
        return stateDict
    
    def loadSaveDict(self,saveDict):
        for i in range(len(self.__modelList)):
            self.__modelList[i].load_state_dict(saveDict[i])
        
        if "optimizer" in saveDict:
            for name in saveDict["optimizer"]:
                self.__optimizerDict[name].load_state_dict(saveDict["optimizer"][name])

        if "scheduler" in saveDict:
            for name in saveDict["scheduler"]:
                self.__schedulerDict[name].load_state_dict(saveDict["scheduler"][name])

                # to reset the learing rate
                self.__schedulerDict[name].step(
                    self.__schedulerDict[name].last_epoch
                )

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
