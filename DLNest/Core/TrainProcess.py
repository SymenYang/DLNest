import multiprocessing
import os
import sys
from multiprocessing import Process
import importlib
import shutil
import json
import torch
from pathlib import Path
from multiprocessing import Process,Pipe,connection,Queue
import gc

try:
    from DLNest.BridgeLayers.InformationLayer import TrainTask
    from DLNest.BridgeLayers.OutputLayers import TrainStdout
except ImportError:
    sys.path.append("..")
    from BridgeLayers.InformationLayer import TrainTask


class TrainProcess(Process):
    def __init__(self,task : TrainTask, showOnScreen = False, commandQueue : Queue = None ):
        super(TrainProcess, self).__init__()
        self.task = task
        self.DEBUG = True
        self.startEpoch = 0
        self.showOnScreen = showOnScreen
        self.commandQueue = commandQueue

    def __loadAModule(self,filePath : Path,name : str):
        if not filePath.is_absolute():
            filePath = Path(self.task.args["root_file_path"]) / filePath
        spec = importlib.util.spec_from_file_location(
            name,
            filePath
        )
        module = importlib.util.module_from_spec(spec)
        dirName = str(filePath.parent)
        if not dirName in sys.path:
            sys.path.append(dirName)
        spec.loader.exec_module(module)
        return module

    def __loadModule(self):
        self.modelModule = self.__loadAModule(self.task.modelFilePath,"Model")
        self.datasetModule = self.__loadAModule(self.task.datasetFilePath,"Dataset")
        self.lifeCycleModule = self.__loadAModule(self.task.lifeCycleFilePath,"LifeCycle")

    def __loadLifeCycle(self):
        self.lifeCycleModule = self.__loadAModule(self.task.lifeCycleFilePath,"LifeCycle")

    def __loadOthers(self):
        self.modelModule = self.__loadAModule(self.task.modelFilePath,"Model")
        self.datasetModule = self.__loadAModule(self.task.datasetFilePath,"Dataset")
        sys.modules["Dataset"] = self.datasetModule

    def __initLifeCycle(self):
        lifeCycleName = self.task.args['life_cycle_name']
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
            self.logDict = self.model.initLog()
            
            # 如果有ckpt_load且不为空，则尝试从文件加载模型
            if "ckpt_load" in self.task.args:
                try:
                    ckpt_file = Path(self.task.args['ckpt_load'])
                    if ckpt_file.is_file():
                        stateDict = torch.load(self.task.args['ckpt_load'])
                        self.model.loadSaveDict(stateDict)
                        self.startEpoch = stateDict['epoch'] + 1
                        self.logDict = stateDict['log_dict']
                except Exception as e:
                    print("[Train Process] [ERROR]",e)
                    print('[Train Process] [ERROR]load ckpt ' , str(self.task.args['ckpt_load']) , 'fail, stop')
                    exit(0)
        else:
            raise Exception("Cannot find model class")

    def __copyAFile(self,filePath : Path, saveDir : Path):
        '''
        若filePath为相对路径，则复制到其对应文件夹
        若filePath为绝对路径，则复制到储存包的根
        '''
        if filePath.is_absolute():
            if filePath.is_dir():
                shutil.copytree(filePath,saveDir / filePath.stem,)
            else:
                shutil.copy(filePath,saveDir / (filePath.stem + filePath.suffix))
        else:
            abFilePath = Path(self.task.args["root_file_path"]) / filePath
            if abFilePath.is_dir():
                shutil.copytree(abFilePath,saveDir / filePath)
            else:
                target = saveDir / filePath
                target_dir = target.parent
                if not target_dir.exists():
                    target_dir.mkdir(parents = True, exist_ok = True)
                shutil.copy(abFilePath,target)

    def __initSave(self):
        # init saving root
        saveRoot = Path(self.task.args["save_root"])
        saveDir = saveRoot / self.task.timestamp
        #print("[Train Process] Saving to " + str(saveDir))
        if saveDir.exists():
            shutil.rmtree(saveDir)
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
        self.__copyAFile(self.task.modelFilePath,saveDir)
        self.__copyAFile(self.task.datasetFilePath,saveDir)
        self.__copyAFile(self.task.lifeCycleFilePath,saveDir)
        for item in self.task.otherFilePaths:
            try:
                self.__copyAFile(item,saveDir)
            except Exception as e:
                print("[Ignored]",e)

        # 重定向输出
        self.outputFile = saveDir / "_output.txt"
        self.outputFP = self.outputFile.open('w')
        if self.showOnScreen:
            self.outputDelegate = TrainStdout(self.outputFP,True,sys.stdout)
        else:
            self.outputDelegate = TrainStdout(self.outputFP)
        sys.stdout = self.outputDelegate
        sys.stderr = self.outputDelegate

        # save args
        argsPath = saveDir / "args.json"
        argsFP = argsPath.open('w')
        self.task.args["_description"] = self.task.description
        self.task.args["_pid"] = os.getpid()
        #self.pid = os.getpid()
        self.task.args["root_file_path"] = str(saveDir)
        json.dump(self.task.args,argsFP,sort_keys=True, indent=4, separators=(',', ':'))
        argsFP.close()
    
    def __saveModel(self,epoch : int):
        pathsNeed2Delete = []

        # save this epoch
        saveDict = self.model.getSaveDict()
        saveDict['epoch'] = epoch
        saveDict['log_dict'] = self.logDict
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
            print("[Train Process] Saved model")

    def __dealCommand(self,command,epoch,iter_now):
        if command == None:
            return
        elif command == "Suspend":
            if self.lifeCycle.BSuspend() != "Skip":
                self.__suspendToDisk(epoch,iter_now)
            self.lifeCycle.ASuspend()
            command = self.commandQueue.get(True)
            self.__dealCommand(command,epoch,iter_now)
        elif command == "LoadFromSuspend":
            if self.lifeCycle.BLoadFromSuspend() != "Skip":
                self.__reloadFromDisk()
            self.lifeCycle.ALoadFromSuspend()
            return

    def __train(self):
        nowEpoch = self.startEpoch
        while True:
            if self.lifeCycle.BOneEpoch() != "Skip":
                for _iter,data in enumerate(self.trainLoader):
                    # get command
                    if self.lifeCycle.BGetCommand() != "Skip":
                        command = None
                        try:
                            command = self.commandQueue.get(False)
                        except Exception as e:
                            ... # no command
                        self.__dealCommand(command,nowEpoch,_iter)
                    self.lifeCycle.AGetCommand(command)

                    # run one step
                    if self.lifeCycle.BModelOneStep() != "Skip":
                        self.model.runOneStep(data,self.logDict,_iter,nowEpoch)
                    self.lifeCycle.AModelOneStep()

                    # visualize
                    if self.lifeCycle.needVisualize(nowEpoch,_iter,self.logDict,self.task.args):
                        if self.lifeCycle.BVisualize() != "Skip":
                            self.model.visualize(epoch = nowEpoch, iter = _iter, log = self.logDict)
                        self.lifeCycle.AVisualize()

                # output in command Line    
                self.lifeCycle.commandLineOutput(nowEpoch,self.logDict,self.task.args)

                # validation
                if self.lifeCycle.needValidation(nowEpoch,self.logDict,self.task.args):
                    if self.lifeCycle.BValidation() != "Skip":
                        self.model.validate(self.valLoader,self.logDict)
                    self.lifeCycle.AValidation()

                # save checkpoint
                if self.lifeCycle.needSaveModel(nowEpoch,self.logDict,self.task.args):
                    if self.lifeCycle.BSaveModel() != "Skip":
                        self.__saveModel(nowEpoch)
                    self.lifeCycle.ASaveModel()
            
            self.lifeCycle.AOneEpoch()
            # break decision
            if self.lifeCycle.needContinueTrain(nowEpoch,self.logDict,self.task.args):
                nowEpoch += 1
            else:
                break

    def __suspendToDisk(self,epoch,iter_now):
        saveDict = self.model.getSaveDict().copy()
        saveDict["epoch"] = epoch
        saveDict["iter"] = iter_now
        saveFile = self.checkpointsDir / "_suspend.ckpt"
        saveName = str(saveFile)

        # del model
        self.lifeCycle.model = None
        self.logDict = None
        self.model = None
        torch.save(saveDict,saveName)
        gc.collect()

        # gc gpu memory
        torch.cuda.empty_cache()
        return True

    def __reloadFromDisk(self):
        ckptFile = self.checkpointsDir / "_suspend.ckpt"
        try:
            self.__initModel()
            stateDict = torch.load(str(ckptFile))
            self.lifeCycle.model = self.model
            self.model.loadSaveDict(stateDict)
            self.startEpoch = stateDict['epoch']
            #self.model.cuda()
            return stateDict["epoch"],stateDict["iter"]
        except Exception as e:
            print("[Train Process] [ERROR]",e)
            print("[Train Process] [ERROR]load ckpt " , str(ckptFile) , 'fail, stop')
            exit(0)

    def run(self):
        if isinstance(self.task.GPUID,list):
            ids = [str(item) for item in self.task.GPUID]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
        elif self.task.GPUID != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.task.GPUID)
        else:
            print("No card has been specified.")
            pass

        self.__loadLifeCycle()
        self.__initLifeCycle()

        self.lifeCycle.trainProcess = self
        if self.lifeCycle.BAll() != "Skip":
            # initialize save folder
            if self.lifeCycle.BSaveInit() != "Skip":
                self.__initSave()
            self.lifeCycle.ASaveInit()

            self.__loadOthers()

            # initialize dataset
            if self.lifeCycle.BDatasetInit() != "Skip":
                self.__initDataset()
                self.lifeCycle.dataset = self.dataset
            self.lifeCycle.ADatasetInit()

            # initialize model
            if self.lifeCycle.BModelInit() != "Skip":
                self.__initModel()
                self.lifeCycle.model = self.model
            self.lifeCycle.AModelInit()

            # train
            if self.lifeCycle.BTrain() != "Skip":
                self.__train()
            self.lifeCycle.ATrain()

        #finalize
        self.lifeCycle.AAll()
        exit(0)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    ta = TrainTask(
        args = {
        "save_root":"/root/code/DLNestTest2/Saves",
        "model_name":"Model",
        "dataset_name":"Dataset",
        "life_cycle_name":"LifeCycle",
        "checkpoint_args":{
            "max_ckpt_in_slow_track":100,
            "dilation_in_slow_track":100,
            "max_ckpt_in_fast_track":2,
            "max_ckpt_in_consistent_track":1
        },
        "root_file_path":"/root/code/DLNestTest2",
        "model_file_path":"./Model/Model.py",
        "dataset_file_path":"./Dataset/Dataset.py",
        "life_cycle_file_path":"./LifeCycle.py",
        "other_file_paths":[],
        "child_jsons":[
            "./model_config.json",
            "./dataset_config.json"
        ]
        },
        description = "none"
    )
    ta.GPUUD = 0
    ta.timestamp = "now"
    TP = TrainProcess(ta)
    TP.start()
    TP.join()