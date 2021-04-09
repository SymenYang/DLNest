from DLNest.Common.DatasetBase import DatasetBase
from DLNest.Common.ModelBase import ModelBase
from DLNest.Common.LifeCycleBase import LifeCycleBase


class LifeCycle(LifeCycleBase):
    def BAll(self):
        self.maxAcc = 0.0

    def needVisualize(self, epoch : int, iter : int, logdict : dict, args : dict):
        return True

    def needValidation(self, epoch : int, logdict : dict, args : dict):
        return True

    def commandLineOutput(self,epoch : int, logdict : dict, args : dict):
        print("Epoch #" + str(epoch) + " finished!")

    def needSaveModel(self, epoch : int, logdict : dict, args : dict):
        return True

    def holdThisCheckpoint(self, epoch : int, logdict : dict, args : dict):
        if logdict["acc"][-1] > self.maxAcc:
            self.maxAcc = logdict["acc"][-1]
            return True
        return False

    def getSaveDict(self):
        return {
            "max_acc" : self.maxAcc
        }

    def loadSaveDict(self,saveDict):
        self.maxAcc = saveDict["max_acc"]

    def needContinueTrain(self, epoch : int, logdict : dict, args : dict):
        if epoch >= args["epochs"]:
            return False
        return True