from DLNest.Information.TaskInformation import TaskInformation
from DLNest.Information.TrainTask import TrainTask
from DLNest.Executor.TaskProcess import TaskProcess
from DLNest.Output.TrainStdout import TrainStdout

from pathlib import Path
import sys

import time
import os

try:
    import torch
    import torch.distributed as dist
except ImportError:
    pass

class TrainProcess(TaskProcess):
    def __init__(self,task : TrainTask,showOnScreen = False):
       """
       if showOnScreen, also output to the stdout
       """
       super(TrainProcess,self).__init__(task)
       self.showOnScreen = showOnScreen
       self.commandQueue = task.commandQueue

    def initOutput(self,rank = -1):
        """
        redirect the output
        """
        os.chdir(self.task.args["root_file_path"]) # Change CWD to the save package
        outputFP = self.task.savePackage.getOutputFile(rank)
        self.outputDelegate = TrainStdout(outputFP,showOnScreen = self.showOnScreen,originalStdout = sys.stdout)
        sys.stdout = self.outputDelegate
        sys.stderr = self.outputDelegate

    def __saveModel(self):
        holdThisCheckpoint = self.lifeCycle.holdThisCheckpoint(self.finishedEpoch,self.logDict,self.task.args)
        self.saveCkpt(holdThisCheckpoint = holdThisCheckpoint)

    def __moveAData(self,data):
        try:
            if "cuda" in dir(data):
                ret_data = data.cuda()
                return ret_data
            elif "to" in dir(data):
                tmp = torch.tensor(1).cuda()
                ret_data = data.to(tmp.device)
                return ret_data
            else:
                return data
        except Exception as e:
            return data

    def __moveData(self,data):
        # move data to the proper location
        if self.envType != "CPU":
            try:
                if isinstance(data,list):
                    for index in range(len(data)):
                        data[index] = self.__moveAData(data[index])
                elif isinstance(data,dict):
                    for key in data:
                        data[key] = self.__moveAData(data[key])
                else:
                    data = self.__moveAData(self, data)
            except Exception as e:
                pass
        return data

    def __train_an_epoch(self,start_epoch : int):
        nowEpoch = start_epoch
        for _iter,data in enumerate(self.trainLoader):
            # run one step
            if self.lifeCycle._BModelOneStep() != "Skip":
                data = self.__moveData(data)
                self.runner._runOneStep(data,self.logDict,_iter,nowEpoch)
            self.lifeCycle._AModelOneStep()

            # visualize
            if self.lifeCycle.needVisualize(nowEpoch,_iter,self.logDict,self.task.args):
                if self.envType == "DDP":
                    if self.rank == 0 and self.lifeCycle._BVisualize() != "Skip":
                        self.runner._visualize(epoch = nowEpoch, iter = _iter, log = self.logDict)
                else:
                    if self.lifeCycle._BVisualize() != "Skip":
                        self.runner._visualize(epoch = nowEpoch, iter = _iter, log = self.logDict)
                self.lifeCycle._AVisualize()


    def __validate(self):
        self.runner._validationInit()
        for _iter,data in enumerate(self.valLoader):
            if self.lifeCycle._BValidateABatch() != "Skip":
                data = self.__moveData(data)
                self.runner._validateABatch(data,_iter)
            self.lifeCycle._AValidateABatch()
        
        if self.lifeCycle._BValidationAnalyze() != "Skip":
            self.runner._validationAnalyze(self.logDict)
        self.lifeCycle._AValidationAnalyze()

    def mainLoop(self):
        try:
            nowEpoch = self.finishedEpoch + 1
            if self.lifeCycle._BTrain() == "Skip":
                self.lifeCycle._ATrain()
                return
            while True:
                if self.lifeCycle._BOneEpoch() != "Skip":
                    if self.envType == "DDP":
                        self.trainLoader.sampler.set_epoch(nowEpoch)

                    self.__train_an_epoch(nowEpoch)                    

                    if self.envType == "DDP":
                        dist.barrier() # Sync before validation

                    self.finishedEpoch = nowEpoch
                    # output in command Line
                    self.lifeCycle._commandLineOutput(self.finishedEpoch,self.logDict,self.task.args)

                    # validation
                    if self.lifeCycle.needValidation(self.finishedEpoch,self.logDict,self.task.args):
                        if self.lifeCycle._BValidation() != "Skip":
                            if "validate" in dir(self.runner):
                                self.runner._validate(self.valLoader,self.logDict)
                            else:
                                self.__validate()
                        self.lifeCycle._AValidation()

                    if self.envType == "DDP":
                        dist.barrier() # Sync before saving

                    #save checkpoint
                    if self.envType == "DDP":
                        if self.rank == 0 and self.lifeCycle._BSaveModel() != "Skip":
                            if self.lifeCycle.needSaveModel(self.finishedEpoch,self.logDict,self.task.args):
                                self.__saveModel()
                    else:
                        if self.lifeCycle._BSaveModel() != "Skip":
                            if self.lifeCycle.needSaveModel(self.finishedEpoch,self.logDict,self.task.args):
                                self.__saveModel()
                    self.lifeCycle._ASaveModel()

                self.lifeCycle._AOneEpoch()
                 # break decision
                if self.lifeCycle.needContinueTrain(self.finishedEpoch,self.logDict,self.task.args):
                    nowEpoch = self.finishedEpoch + 1
                else:
                    break
        except (Exception,SystemExit) as e:
            self.lifeCycle._trainAborting(e)
        else:
            # After Train
            self.lifeCycle._ATrain()

    def loadCkpt(self):
        super().loadCkpt()
        if self.task.loadCkpt:
            if self.task.checkpointID != -1:
                # -1 means the last one,which is default option. != -1 needs to set the ckptID rather than default
                self.task.savePackage.setCkptID(self.task.checkpointID + 1)

if __name__ == "__main__":
    TT = TrainTask.fromConfigFile("/root/code/DLNestTest/root_config.json",devices = [0,1,2,3],noSave = True,DDP = True)
    # TT = TrainTask.fromRecord("/root/code/DLNestTest/Saves/2021-03-29_16-04-46_146",checkpointID = 5,devices = [0,1,2,3],DDP = False)
    TP = TrainProcess(TT,True)
    TP.start()
    try:
        while True:
            pass
    except KeyboardInterrupt:
        TP.terminate()
    TP.join()
    print("terminate")
