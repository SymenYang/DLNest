from abc import ABCMeta, abstractmethod
from .DatasetBase import DatasetBase
from .ModelBase import ModelBase
import traceback

class LifeCycleBase:
    def __init__(self,model : ModelBase = None,dataset : DatasetBase = None, taskProcess = None, rank : int = -1):
        self.model = model
        self.dataset = dataset
        self.taskProcess = taskProcess
        self.rank = rank
    
    def BAll(self):
        pass

    def BDatasetInit(self):
        pass

    def ADatasetInit(self):
        pass

    def BModelInit(self):
        pass

    def AModelInit(self):
        pass

    def BTrain(self):
        pass

    def ATrain(self):
        pass

    def BOneEpoch(self):
        pass

    def AOneEpoch(self):
        pass

    def BGetCommand(self):
        pass

    def AGetCommand(self,command):
        pass

    def BSuspend(self):
        pass

    def ASuspend(self):
        pass

    def BLoadFromSuspend(self):
        pass

    def ALoadFromSuspend(self):
        pass

    def BModelOneStep(self):
        pass

    def AModelOneStep(self):
        pass

    @abstractmethod
    def needVisualize(self, epoch : int, iter : int, logdict : dict, args : dict):
        return False
    
    def BVisualize(self):
        pass

    def AVisualize(self):
        pass

    @abstractmethod
    def needValidation(self, epoch : int, logdict : dict, args : dict):
        return False

    def BValidation(self):
        pass

    def AValidation(self):
        pass

    def commandLineOutput(self,epoch : int, logdict : dict, args : dict):
        print("Epoch #" + str(epoch) + " finished!")

    @abstractmethod
    def needSaveModel(self, epoch : int, logdict : dict, args : dict):
        return True

    def BSaveModel(self):
        pass

    def ASaveModel(self):
        pass

    @abstractmethod
    def holdThisCheckpoint(self, epoch : int, logdict : dict, args : dict):
        return False

    @abstractmethod
    def needContinueTrain(self, epoch : int, logdict : dict, args : dict):
        return False
    
    def getSaveDict(self):
        return {}

    def loadSaveDict(self,saveDict):
        pass

    def AAll(self):
        pass

    def TrainAborting(self,exception : Exception):
        traceback.print_exc()