from abc import ABCMeta, abstractmethod

class ModelBase:
    def __init__(self,_envType : str,rank = -1):
        self._envType = _envType
        self.rank = rank
        pass
    
    @abstractmethod
    def init(self,args : dict, datasetInfo : dict = None):
        self.args = args
        pass
    
    @abstractmethod
    def initOptimizer(self):
        pass
    
    @abstractmethod
    def DDPOperation(self,rank : int):
        pass

    def afterDDP(self, rank : int):
        """
        For SyncBN or something else.
        """
        pass

    @abstractmethod
    def initLog(self):
        return {}
    
    @abstractmethod
    def getSaveDict(self):
        return {}
    
    @abstractmethod
    def loadSaveDict(self,saveDict):
        pass

    @abstractmethod
    def runOneStep(self,data,log : dict, iter : int, epoch : int):
        pass

    def visualize(self,log : dict, iter : int, epoch : int):
        pass

    def validate(self,valLoader, log : dict):
        pass