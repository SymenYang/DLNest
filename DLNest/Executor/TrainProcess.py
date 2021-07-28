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

    def mainLoop(self):
        try:
            nowEpoch = self.finishedEpoch + 1
            if self.lifeCycle.BTrain() == "Skip":
                self.lifeCycle.ATrain()
                return
            while True:
                if self.lifeCycle.BOneEpoch() != "Skip":
                    if self.envType == "DDP":
                        self.trainLoader.sampler.set_epoch(nowEpoch)
                    for _iter,data in enumerate(self.trainLoader):
                        # run one step
                        if self.lifeCycle.BModelOneStep() != "Skip":
                            # move data to the proper location
                            if self.envType != "CPU":
                                if isinstance(data,torch.Tensor):
                                    data = data.cuda()
                                elif isinstance(data,list):
                                    for index in range(len(data)):
                                        if isinstance(data[index],torch.Tensor):
                                            data[index] = data[index].cuda()
                                elif isinstance(data,dict):
                                    for key in data:
                                        if isinstance(data[key],torch.Tensor):
                                            data[key] = data[key].cuda() 
                            self.model.runOneStep(data,self.logDict,_iter,nowEpoch)
                        self.lifeCycle.AModelOneStep()

                        # visualize
                        if self.lifeCycle.needVisualize(nowEpoch,_iter,self.logDict,self.task.args):
                            if self.envType == "DDP":
                                if self.rank == 0 and self.lifeCycle.BVisualize() != "Skip":
                                    self.model.visualize(epoch = nowEpoch, iter = _iter, log = self.logDict)
                            else:
                                if self.lifeCycle.BVisualize() != "Skip":
                                    self.model.visualize(epoch = nowEpoch, iter = _iter, log = self.logDict)
                            self.lifeCycle.AVisualize()

                    if self.envType == "DDP":
                        dist.barrier() # Sync before validation

                    self.finishedEpoch = nowEpoch
                    # output in command Line
                    self.lifeCycle.commandLineOutput(self.finishedEpoch,self.logDict,self.task.args)

                    # validation
                    if self.lifeCycle.needValidation(self.finishedEpoch,self.logDict,self.task.args):
                        if self.lifeCycle.BValidation() != "Skip":
                            self.model.validate(self.valLoader,self.logDict)
                        self.lifeCycle.AValidation()

                    if self.envType == "DDP":
                        dist.barrier() # Sync before saving

                    #save checkpoint
                    if self.envType == "DDP":
                        if self.rank == 0 and self.lifeCycle.BSaveModel() != "Skip":
                            if self.lifeCycle.needSaveModel(self.finishedEpoch,self.logDict,self.task.args):
                                self.__saveModel()
                    else:
                        if self.lifeCycle.BSaveModel() != "Skip":
                            if self.lifeCycle.needSaveModel(self.finishedEpoch,self.logDict,self.task.args):
                                self.__saveModel()
                    self.lifeCycle.ASaveModel()

                self.lifeCycle.AOneEpoch()
                 # break decision
                if self.lifeCycle.needContinueTrain(self.finishedEpoch,self.logDict,self.task.args):
                    nowEpoch = self.finishedEpoch + 1
                else:
                    break
        except Exception as e:
            self.lifeCycle.TrainAborting(e)
        
        # After Train
        self.lifeCycle.ATrain()

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
