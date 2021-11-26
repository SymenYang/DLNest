try:
    import torch
except ImportError:
    pass
from functools import wraps
import logging
from DLNest.Plugins.Utils.CheckPlugins import checkPlugins,checkDictOutputPlugins

class DatasetBase:
    def __init__(self,_envType : str = "CPU", args : dict = {}, plugins : list = []):
        self._envType = _envType
        self._args = args
        self._plugins = plugins

    def getArgs(self):
        return self._args

    @checkPlugins
    def _datasetInit(self, args : dict):
        return self.init(args = args)

    def init(self,args : dict):
        """
        input:
            args : dict
        output:
            dict to runner
            train loader
            val loader
        """
        return {},None,None

    def getSampler(self, dataset):
        if self._envType == "DDP":
            return torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            return None

    @checkDictOutputPlugins
    def _getSaveDict(self):
        return self.getSaveDict()

    def getSaveDict(self):
        return {}
    
    @checkPlugins
    def _loadSaveDict(self, saveDict):
        return self.loadSaveDict(saveDict = saveDict)

    def loadSaveDict(self,saveDict):
        pass