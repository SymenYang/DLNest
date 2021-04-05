from abc import ABCMeta, abstractmethod

class DatasetBase:
    def __init__(self,args : dict):
        self.args = args
    
    @abstractmethod
    def afterInit(self,DDP : bool = False):
        """
        input:
            DDP : bool, True to notify using DDP sampler
        output:
            dict to model
            train loader
            val loader
        """
        return {},None,None
    
    def getSaveDict(self):
        return {}
    
    def loadSaveDict(self,saveDict):
        pass