from DLNest.Information.TaskInformation import TaskInformation

import multiprocessing
import os
import sys
from multiprocessing import Process
import importlib
import shutil
import json
import random
from pathlib import Path
import numpy as np

from abc import ABCMeta, abstractmethod
try:
    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    import torch.multiprocessing as mp
    USINGTORCH = True
except ImportError:
    USINGTORCH = False

class TaskProcess(Process):
    def __init__(self,task : TaskInformation):
        super(TaskProcess,self).__init__()
        self.task = task
        self.finishedEpoch = -1
        self.rank = -1
        self.worldSize = -1
    
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

    def __loadLifeCycle(self):
        self.lifeCycleModule = self.__loadAModule(Path(self.task.args["life_cycle_file_path"]),"LifeCycle")

    def __initLifeCycle(self,rank : int = -1):
        lifeCycleName = self.task.args['life_cycle_name']
        if lifeCycleName in dir(self.lifeCycleModule):
            self.lifeCycle = self.lifeCycleModule.__getattribute__(lifeCycleName)(taskProcess = self,rank = rank, plugins = self.plugins)
            if self.task.loadCkpt:
                self.lifeCycle._loadSaveDict(self.stateDict["life_cycle"])
        else:
            raise Exception("Cannot find lifeCycle class")

    def __loadAPlugin(self,pluginName : str):
        pluginPath = Path(__file__).parent.parent / "Plugins" / (pluginName + '.py')
        tmpPath = Path(pluginName)
        if tmpPath.is_absolute():
            pluginPath = tmpPath
            pluginName = tmpPath.name
        pluginModule = self.__loadAModule(filePath = pluginPath,name = pluginName)
        pluginClass = pluginModule.__getattribute__("DLNestPlugin")
        return pluginClass

    def __loadPlugins(self):
        if not "plugins" in self.task.args:
            return []
        pluginNames = self.task.args["plugins"]
        pluginList = []
        for name in pluginNames:
            pluginList.append(self.__loadAPlugin(name))
        self.plugins = pluginList

    def __loadRunnerAndDataset(self):
        runnerPath = Path(self.task.args["runner_file_path"] if "runner_file_path" in self.task.args else self.task.args["model_file_path"]) # need to be deprecated
        datasetPath = Path(self.task.args["dataset_file_path"])
        self.runnerModule = self.__loadAModule(runnerPath,runnerPath.stem)
        self.datasetModule = self.__loadAModule(datasetPath,datasetPath.stem)
        sys.modules[datasetPath.stem] = self.datasetModule # Why?

    def __initDataset(self):
        datasetName = self.task.args['dataset_name']
        if datasetName in dir(self.datasetModule):
            datasetClass = self.datasetModule.__getattribute__(datasetName)
            self.dataset = datasetClass(_envType = self.envType,args = self.task.args, plugins = self.plugins)
            self.datasetInfo,self.trainLoader,self.valLoader = self.dataset._datasetInit(self.task.args)
            # load from ckpt is needed.
            if self.task.loadCkpt:
                self.dataset._loadSaveDict(self.stateDict["dataset"])
        else:
            raise Exception("Cannot find dataset class")

    def __initRunner(self):
        runnerName = self.task.args['runner_name'] if "runner_name" in self.task.args else self.task.args["model_name"] # need to be deprecated
        if runnerName in dir(self.runnerModule):
            runnerClass = self.runnerModule.__getattribute__(runnerName)
            self.runner = runnerClass(_envType = self.envType,args = self.task.args, rank = self.rank,worldSize = self.worldSize, plugins = self.plugins)
            self.runner._runnerInit(self.task.args,self.datasetInfo)
            if self.envType != "DDP":
                self.runner._initOptimizer()
            else:
                self.runner.DDPOperation(rank = self.rank)
                self.runner._initOptimizer()
                self.runner.afterDDP(rank = self.rank)

            # load from ckpt is needed.
            if self.task.loadCkpt:
                self.runner._loadSaveDict(self.stateDict["runner"] if "runner" in self.stateDict else self.stateDict["model"])
            else:
                # if load from ckpt, logDict has been loaded in self.loadCkpt
                self.logDict = self.runner._initLog()

    def checkDeviceEnviroment(self):
        """
        make correct environ params and 
        return "CPU","GPU","GPUs","DDP"
        """
        assert len(self.task.devices) > 0
        if self.task.devices[0] == -1:
            return "CPU"

        ids = [str(item) for item in self.task.devices]
        assert not ("-1" in ids) #no CPU in the devices list. Preventing world size error
        
        if self.task.DDP:
            assert USINGTORCH
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
            self.deviceNum = len(ids)
            return "DDP"
        elif self.task.multiGPU:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
            return "GPUs"
        else:
            torch.cuda.set_device(self.task.devices[0])
            os.environ["CUDA_VISIBLE_DEVICES"] = ids[0]
            return "GPU"
    
    def setupSeed(self):
        seed = self.seed
        np.random.seed(seed)
        random.seed(seed)
        if USINGTORCH:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

    def initBeforeDDP(self):
        self.seed = random.randint(0,2147483647)

    def initAfterDDP(self,rank,worldSize):
        os.environ["MASTER_ADDR"] = self.task.address
        os.environ["MASTER_PORT"] = self.task.port
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank = rank, world_size = worldSize)
        self.rank = rank
        self.worldSize = worldSize
        self.setupSeed()
    
    def runDDP(self,rank):
        self.initAfterDDP(rank,self.deviceNum)
        self.initOutput(rank = rank)
        self.loadCkpt()
        if self.initModules(rank = rank) != "Skip":
            dist.barrier() # start together
            self.mainLoop()

        self.lifeCycle._AAll()
        dist.destroy_process_group()
        exit(0)

    @abstractmethod
    def mainLoop(self):
        """
        needs to be override by other subprocess.
        """
        pass
    
    @abstractmethod
    def initOutput(self,rank = -1):
        """
        needs to be override by other subprocess.
        """
        pass

    def loadCkpt(self):
        """
        If task have ckpt to load, load it to the pointed device.
        """
        deviceStr = ""
        if self.task.devices[0] == -1:
            deviceStr = "cpu"
        else:
            deviceStr = "cuda"
        if self.task.loadCkpt:
            self.stateDict = self.task.savePackage.getStateDict(self.task.checkpointID,deviceStr)
            self.logDict = self.stateDict["log_dict"]
            self.finishedEpoch = self.stateDict["finished_epoch"]

    def saveCkpt(self,otherDict : dict = {}, holdThisCheckpoint = False):
        """
        Save the states of life cycle, dataset and runner.
        """
        stateDict = {}
        stateDict["life_cycle"] = self.lifeCycle._getSaveDict()
        stateDict["dataset"] = self.dataset._getSaveDict()
        stateDict["runner"] = self.runner._getSaveDict()
        stateDict["log_dict"] = self.logDict
        stateDict["finished_epoch"] = self.finishedEpoch
        for key in otherDict:
            stateDict[key] = otherDict[key]
        self.task.savePackage.saveACheckpoint(stateDict,holdThisCheckpoint=holdThisCheckpoint)

    def initModules(self,rank : int = -1):
        """
        Init all modules
        """
        self.__loadLifeCycle()
        self.__loadPlugins()
        self.__initLifeCycle(rank = rank)

        self.__loadRunnerAndDataset()

        if self.lifeCycle._BAll() != "Skip":
            if self.lifeCycle._BDatasetInit() != "Skip":
                self.__initDataset()
                self.lifeCycle.dataset = self.dataset
            self.lifeCycle._ADatasetInit()
        
            if self.lifeCycle._BModelInit() != "Skip":
                self.__initRunner()
                self.lifeCycle.runner = self.runner
            self.lifeCycle._AModelInit()
            return
        else:
            return "Skip"

    def run(self):
        self.envType = self.checkDeviceEnviroment()
        if self.envType == "DDP":
            # run DDP
            self.ppid = os.getpid()
            self.initBeforeDDP()
            context = mp.spawn(
                self.runDDP,
                nprocs=self.deviceNum,
                join=False,
                daemon = True
            )
            self.initOutput()
            while not context.join():
                pass
        else:
            self.initOutput()
            self.loadCkpt()
            if self.initModules() != "Skip":
                self.mainLoop()
            self.lifeCycle._AAll()
