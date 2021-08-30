from DLNest.Common.DatasetBase import DatasetBase
from DLNest.Common.RunnerBase import RunnerBase
from DLNest.Common.LifeCycleBase import LifeCycleBase


class LifeCycle(LifeCycleBase):
    def needVisualize(self, epoch : int, iter : int, logdict : dict, args : dict):
        return False

    def needValidation(self, epoch : int, logdict : dict, args : dict):
        return False

    def commandLineOutput(self,epoch : int, logdict : dict, args : dict):
        print("Epoch #" + str(epoch) + " finished!")

    def needSaveModel(self, epoch : int, logdict : dict, args : dict):
        return False

    def holdThisCheckpoint(self, epoch : int, logdict : dict, args : dict):
        return False

    def needContinueTrain(self, epoch : int, logdict : dict, args : dict):
        return False