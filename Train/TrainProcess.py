import multiprocessing
import os
import sys
from multiprocessing import Process
import importlib
import shutil
import json
import torch


try:
    from Task import Task
except ImportError:
    from .Task import Task

# debug
from pathlib import Path

class TrainProcess(Process):
    def __init__(self,task : Task):
        super(TrainProcess, self).__init__()
        self.task = task
        self.DEBUG = True

    def __loadAModule(self,filePath : Path,name : str):
        spec = importlib.util.spec_from_file_location(
            name,
            filePath
        )
        module = importlib.util.module_from_spec(spec)
        dirName = str(filePath.parents[0])
        if not dirName in sys.path:
            sys.path.append(dirName)
        spec.loader.exec_module(module)
        return module

    def __loadModule(self):
        self.modelModule = self.__loadAModule(self.task.modelFilePath,"Model")
        self.datasetModule = self.__loadAModule(self.task.datasetFilePath,"Dataset")
        self.lifeCycleModule = self.__loadAModule(self.task.lifeCycleFilePath,"Lifecycle")

    def __initLifeCycle(self):
        lifeCycleName = self.task.args['lifeCycle_name']
        if lifeCycleName in dir(self.lifeCycleModule):
            self.lifeCycle = self.lifeCycleModule.__getattribute__(lifeCycleName)()
        else:
            raise Exception("Cannot find lifeCycle class")

    def __initDataset(self):
        datasetName = self.task.args['dataset_name']
        if datasetName in dir(self.datasetModule):
            datasetClass = self.datasetModule.__getattribute__(datasetName)
            self.dataset = datasetClass(self.task.args)
            self.datasetInfo,self.trainLoader,self.valLoader = self.dataset.afterInit()
        else:
            raise Exception("Cannot find dataset class")

    def __initModel(self):
        modelName = self.task.args['model_name']
        if modelName in dir(self.modelModule):
            modelClass = self.modelModule.__getattribute__(modelName)
            self.model = modelClass(self.task.args,self.datasetInfo)
        else:
            raise Exception("Cannot find model class")
    
    def __initSave(self):
        # init saving root
        saveRoot = Path(self.task.args["save_root"])
        saveDir = saveRoot / self.task.timestamp
        print("Saving to " + str(saveDir))
        saveDir.mkdir(parents=True,exist_ok=True)

        # checkpoints saving dir
        self.checkpointsDir = saveDir / "Checkpoints"
        self.checkpointsDir.mkdir(parents=True,exist_ok=True)

        # checkpoint lists
        ckptArgs = self.task.args["checkpoint_args"]
        self.ckptSlow = []
        self.maxCkptSlow = ckptArgs["max_ckpt_in_slow_track"]
        self.slowDilation = ckptArgs["dilation_in_slow_track"]
        self.ckptFast = []
        self.maxCkptFast = ckptArgs["max_ckpt_in_fast_track"]
        self.ckptConsistent = []
        self.maxCkptConsistent = ckptArgs["max_ckpt_in_consistent_track"]

        # copy python files into dir
        modelTargetPath = saveDir / "_Model.py"
        datasetTargetPath = saveDir / "_Dataset.py"
        lifeCycleTargetPath = saveDir / "_Lifecycle.py"
        shutil.copy(self.task.modelFilePath,modelTargetPath)
        shutil.copy(self.task.datasetFilePath,datasetTargetPath)
        shutil.copy(self.task.lifeCycleFilePath,lifeCycleTargetPath)
        for item in self.task.otherFilePaths:
            if item.is_file():
                src = str(item)
                dstPath = saveDir / (item.stem + item.suffix)
                dst = str(dstPath)
                shutil.copy(src,dst)
        
        # 重定向输出
        self.outputFile = saveDir / "_output.txt"
        self.outputFP = self.outputFile.open('w')
        sys.stdout = self.outputFP
        sys.stderr = self.outputFP

        # save args
        argsPath = saveDir / "args.json"
        argsFP = argsPath.open('w')
        self.task.args["_description"] = self.task.description
        self.task.args["_pid"] = os.getpid()
        json.dump(self.task.args,argsFP,sort_keys=True, indent=4, separators=(',', ':'))
        argsFP.close()
    
    def __saveModel(self,epoch : int):
        pathsNeed2Delete = []

        # save this epoch
        saveDict = self.model.getSaveDict()
        saveDict['epoch'] = epoch
        saveFile = self.checkpointsDir / ("epoch_" + str(epoch) + ".ckpt")
        saveName = str(saveFile)
        torch.save(saveDict,saveName)

        # append in fast track
        self.ckptFast.append(saveFile)
        if len(self.ckptFast) > self.maxCkptFast:
            w2dPath = self.ckptFast.pop(0)
            pathsNeed2Delete.append(w2dPath)
        
        # append in slow track
        if epoch % self.slowDilation == 0:
            self.ckptSlow.append(saveFile)
            if len(self.ckptSlow) > self.maxCkptSlow:
                w2dPath = self.ckptSlow.pop(0)
                if not w2dPath in pathsNeed2Delete:
                    pathsNeed2Delete.append(w2dPath)
        
        # append in consistent track
        if self.lifeCycle.holdThisCheckpoint(epoch,self.logDict,self.task.args):
            self.ckptConsistent.append(saveFile)
            if len(self.ckptConsistent) > self.maxCkptConsistent:
                w2dPath = self.ckptConsistent.pop(0)
                if not w2dPath in pathsNeed2Delete:
                    pathsNeed2Delete.append(w2dPath)
        
        # delete useless checkpoints on disk
        for item in pathsNeed2Delete:
            if not (item in self.ckptFast or
                    item in self.ckptSlow or
                    item in self.ckptConsistent):
                item.unlink()

        if self.DEBUG:
            print("Saved model")

    def __train(self):
        nowEpoch = 0
        while True:
            for _iter,data in enumerate(self.trainLoader):
                # run one step
                self.lifeCycle.BModelOneStep()
                self.model.runOneStep(data,self.logDict,_iter,nowEpoch)
                self.lifeCycle.AModelOneStep()
                
                # visualize
                if self.lifeCycle.needVisualize(nowEpoch,_iter,self.logDict,self.task.args):
                    self.lifeCycle.BVisualize()
                    self.model.visualize()
                    self.lifeCycle.AVisualize()

            # output in command Line    
            self.lifeCycle.commandLineOutput(nowEpoch,self.logDict,self.task.args)

            # save checkpoint
            if self.lifeCycle.needSaveModel(nowEpoch,self.logDict,self.task.args):
                self.lifeCycle.BSaveModel()
                self.__saveModel(nowEpoch)
                self.lifeCycle.ASaveModel()

            # validation
            if self.lifeCycle.needValidation(nowEpoch,self.logDict,self.task.args):
                self.lifeCycle.BValidation()
                self.model.validate(self.valLoader)
                self.lifeCycle.AVisualize()

            # break decision
            if self.lifeCycle.needContinueTrain(nowEpoch,self.logDict,self.task.args):
                nowEpoch += 1
            else:
                break

    def run(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.task.GPUID)
        if self.DEBUG:
            print("Running on GPU ",self.task.GPUID,self.task.description)

        self.__loadModule()
        self.__initLifeCycle()

        self.lifeCycle.trainProcess = self
        self.lifeCycle.BAll()
        
        # initialize dataset
        self.lifeCycle.BDatasetInit()
        self.__initDataset()
        self.lifeCycle.dataset = self.dataset
        self.lifeCycle.ADatasetInit()
        
        # initialize model
        self.lifeCycle.BModelInit()
        self.__initModel()
        self.lifeCycle.model = self.model
        self.logDict = self.model.initLog()
        self.lifeCycle.AModelInit()

        # initialize save folder
        self.lifeCycle.BSaveInit()
        self.__initSave()
        self.lifeCycle.ASaveInit()

        # train
        self.lifeCycle.BTrain()
        self.__train()
        self.lifeCycle.ATrain()

        #finalize
        self.lifeCycle.AAll()
        exit(0)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    ta = Task(
        modelFilePath = Path("/root/code/DLNestTest/TestModel.py"),
        datasetFilePath = Path("/root/code/DLNestTest/Dataset.py"),
        lifeCycleFilePath = Path("/root/code/DLNestTest/LifeCycle.py"),
        otherFilePaths = [
            Path("/root/code/DLNestTest/DatasetBase.py"),
            Path("/root/code/DLNestTest/__init__.py")
        ],
        args = {
            "save_root" : "/root/code/DLNestTest/Saves",
            "model_name" : "ModelTest",
            "dataset_name" : "DatasetTest",
            "lifeCycle_name" : "LifeCycleTest",
            "checkpoint_args" : {
                "max_ckpt_in_slow_track" : 3,
                "dilation_in_slow_track" : 2,
                "max_ckpt_in_fast_track" : 2,
                "max_ckpt_in_consistent_track" : 2
            }
        },
        GPUID = 0,
        timestamp = "now",
        description = "none"
    )
    TP = TrainProcess(ta)
    TP.start()
    TP.join()