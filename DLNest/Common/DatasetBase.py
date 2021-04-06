from abc import ABCMeta, abstractmethod
try:
    import torch
except ImportError:
    pass

class DatasetBase:
    def __init__(self,_envType : str):
        self._envType = _envType

    @abstractmethod
    def init(self,args : dict):
        """
        input:
            args : dict
        output:
            dict to model
            train loader
            val loader
        """
        return {},None,None
    
    def getSampler(self, dataset):
        if self._envType == "DDP":
            return torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            return None

    def getSaveDict(self):
        return {}
    
    def loadSaveDict(self,saveDict):
        pass