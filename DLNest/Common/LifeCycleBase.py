from abc import ABCMeta, abstractmethod
try:
    from .DatasetBase import DatasetBase
    from .RunnerBase import RunnerBase
except ImportError:
    from DLNest.Common.DatasetBase import DatasetBase
    from DLNest.Common.RunnerBase import RunnerBase
from DLNest.Plugins.Utils.CheckPlugins import checkPlugins
import traceback
from functools import wraps
import logging

class LifeCycleBase:
    def __init__(self,runner : RunnerBase = None,dataset : DatasetBase = None, taskProcess = None, plugins : list = [], status = None):
        self.runner = runner
        self.dataset = dataset
        self.taskProcess = taskProcess
        self._plugins = plugins
        self._status = status

    # for backward compatibility
    @property
    def rank(self):
        if not "_warned_rank" in dir(self):
            self._warned_rank = True
            print("LifeCycle.rank is deprecated, please use LifeCycle.status.rank")
        return self._status.rank

    def getArgs(self):
        return self.taskProcess.task.args

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

    def BValidateABatch(self):
        pass

    def AValidateABatch(self):
        pass

    def BValidationAnalyze(self):
        pass

    def AValidationAnalyze(self):
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

    def trainAborting(self,exception : Exception):
        traceback.print_exc()

    @checkPlugins
    def _BAll(self):
        return self.BAll()

    @checkPlugins
    def _BDatasetInit(self):
        return self.BDatasetInit()

    @checkPlugins
    def _ADatasetInit(self):
        return self.ADatasetInit()

    @checkPlugins
    def _BModelInit(self):
        return self.BModelInit()

    @checkPlugins
    def _AModelInit(self):
        return self.AModelInit()

    @checkPlugins
    def _BTrain(self):
        return self.BTrain()

    @checkPlugins
    def _ATrain(self):
        return self.ATrain()

    @checkPlugins
    def _BOneEpoch(self):
        return self.BOneEpoch()

    @checkPlugins
    def _AOneEpoch(self):
        return self.AOneEpoch()

    @checkPlugins
    def _BGetCommand(self):
        return self.BGetCommand()

    @checkPlugins
    def _AGetCommand(self,command):
        return self.AGetCommand(command = command)

    @checkPlugins
    def _BSuspend(self):
        return self.BSuspend()

    @checkPlugins
    def _ASuspend(self):
        return self.ASuspend()

    @checkPlugins
    def _BLoadFromSuspend(self):
        return self.BLoadFromSuspend()

    @checkPlugins
    def _ALoadFromSuspend(self):
        return self.ALoadFromSuspend()

    @checkPlugins
    def _BModelOneStep(self):
        return self.BModelOneStep()

    @checkPlugins
    def _AModelOneStep(self):
        return self.AModelOneStep()

    @checkPlugins
    def _BVisualize(self):
        return self.BVisualize()

    @checkPlugins
    def _AVisualize(self):
        return self.AVisualize()

    @checkPlugins
    def _BValidation(self):
        return self.BValidation()
    
    @checkPlugins
    def _BValidateABatch(self):
        return self.BValidateABatch()
    
    @checkPlugins
    def _AValidateABatch(self):
        return self.AValidateABatch()
    
    @checkPlugins
    def _BValidationAnalyze(self):
        return self.BValidationAnalyze()
    
    @checkPlugins
    def _AValidationAnalyze(self):
        return self.AValidationAnalyze()
    
    @checkPlugins
    def _AValidation(self):
        return self.AValidation()
    
    @checkPlugins
    def _commandLineOutput(self,epoch : int, logdict : dict, args : dict):
        return self.commandLineOutput(epoch = epoch, logdict = logdict, args = args)
    
    @checkPlugins
    def _BSaveModel(self):
        return self.BSaveModel()
    
    @checkPlugins
    def _ASaveModel(self):
        return self.ASaveModel()

    @checkPlugins    
    def _getSaveDict(self):
        return self.getSaveDict()
    
    @checkPlugins
    def _loadSaveDict(self,saveDict):
        return self.loadSaveDict(saveDict)

    @checkPlugins
    def _AAll(self):
        return self.AAll()

    @checkPlugins
    def _trainAborting(self,exception : Exception):
        return self.trainAborting(exception = exception)