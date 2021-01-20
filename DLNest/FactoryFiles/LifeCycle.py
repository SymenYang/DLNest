from DLNest.Common.DatasetBase import DatasetBase
from DLNest.Common.ModelBase import ModelBase
from DLNest.Common.LifeCycleBase import LifeCycleBase


class LifeCycle(LifeCycleBase):
    def __init__(self,model : ModelBase = None,dataset : DatasetBase = None, trainProcess = None, analyzeProcess = None):
        self.model = model
        self.dataset = dataset
        self.trainProcess = trainProcess
        self.analyzeProcess = analyzeProcess

    def needVisualize(self, epoch : int, iter : int, logdict : dict, args : dict):
        return False

    def needValidation(self, epoch : int, logdict : dict, args : dict):
        return False

    def commandLineOutput(self,epoch : int, logdict : dict, args : dict):
        print("Epoch #" + str(epoch) + " finished!")

    def needSaveModel(self, epoch : int, logdict : dict, args : dict):
        return True

    def holdThisCheckpoint(self, epoch : int, logdict : dict, args : dict):
        return False

    def needContinueTrain(self, epoch : int, logdict : dict, args : dict):
        return False