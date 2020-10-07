import os
import sys
import importlib
import shutil
import json
import torch
from pathlib import Path
import multiprocessing
from multiprocessing import Process
from prompt_toolkit import PromptSession,HTML
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

try:
    from AnalyzeTask import AnalyzeTask
except ImportError:
    from .AnalyzeTask import AnalyzeTask

class AnalyzeRunner:
    def __init__(self,args : dict, model , dataset):
        self.args = args
        self.model = model
        self.dataset = dataset

class AnalyzerProcess(Process):
    def __init__(self,task : AnalyzeTask):
        super(AnalyzerProcess, self).__init__()
        self.task = task
        # 如果两个path不是目录的话，直接返回
        if not (self.task.recordPath.is_dir() and self.task.scriptPath.is_dir()):
            raise Exception("Wrong directory arguments.")
            return

        self.file = Path(os.path.realpath(__file__))
        self.parentDir = self.file.parent

        self.DEBUG = True

        # 用以在子进程中操作主进程的stdin
        self.stdin = None

    def __loadAModule(self,filePath : Path,name : str):
        # 按照文件名加载一个模块
        spec = importlib.util.spec_from_file_location(
            name,
            filePath
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def __loadModules(self):
        # 将Common中的三个Base计入sys.path
        commonDirName = str(self.parentDir.parent / "Common")
        if not commonDirName in sys.path:
            sys.path.append(commonDirName)
        
        # 将保存文件夹内的文件计入sys.path
        recordDirName = str(self.task.recordPath)
        if not recordDirName in sys.path:
            sys.path.append(recordDirName)
        
        self.modelModule = self.__loadAModule(self.task.recordPath / "_Model.py","Model")
        self.datasetModule = self.__loadAModule(self.task.recordPath / "_Dataset.py","Dataset")
        self.lifeCycleModule = self.__loadAModule(self.task.recordPath / "_LifeCycle.py","LifeCycle")
        
        # 加载args
        with (self.task.recordPath / "args.json").open('r') as fp:
            self.args = json.load(fp)

    def __setCheckpoint(self):
        # 找到checkpoint的文件，并加入args
        checkpointID = self.task.checkpointID
        checkpointFilePath = Path(self.task.recordPath / "Checkpoints" / ("epoch_" + str(checkpointID) + ".ckpt"))
        if not checkpointFilePath.is_file():
            raise Exception("Wrong checkpoint id.")
        self.args['ckpt_load'] = str(checkpointFilePath)

    def __initLifeCycle(self):
        lifeCycleName = self.args['life_cycle_name']
        if lifeCycleName in dir(self.lifeCycleModule):
            self.lifeCycle = self.lifeCycleModule.__getattribute__(lifeCycleName)()
        else:
            raise Exception("Cannot find lifeCycle class")

    def __initDataset(self):
        datasetName = self.args['dataset_name']
        if datasetName in dir(self.datasetModule):
            datasetClass = self.datasetModule.__getattribute__(datasetName)
            self.dataset = datasetClass(self.args)
            self.datasetInfo,self.trainLoader,self.valLoader = self.dataset.afterInit()
        else:
            raise Exception("Cannot find dataset class")

    def __initModel(self):
        modelName = self.args['model_name']
        if modelName in dir(self.modelModule):
            modelClass = self.modelModule.__getattribute__(modelName)
            self.model = modelClass(self.args,self.datasetInfo)
            self.logDict = self.model.initLog()
            
            # 如果有ckpt_load且不为空，则尝试从文件加载模型
            if "ckpt_load" in self.args:
                try:
                    ckpt_file = Path(self.args['ckpt_load'])
                    if ckpt_file.is_file():
                        stateDict = torch.load(self.args['ckpt_load'])
                        self.model.loadSaveDict(stateDict)
                        self.startEpoch = stateDict['epoch']
                except Exception as e:
                    print(e)
                    print('load ckpt ' , str(self.args['ckpt_load']) , 'fail, stop')
                    exit(0)
        else:
            raise Exception("Cannot find model class")

    def __initAll(self):
        self.__loadModules()
        self.__setCheckpoint()
        self.__initLifeCycle()

        self.lifeCycle.analyzeProcess = self
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
        self.lifeCycle.AModelInit()

        # 设置runner，作为跑脚本的主体，也就是self
        self.runner = AnalyzeRunner(self.args,self.model,self.dataset)

    def __startTest(self,commandWordList):
        command = commandWordList[1]
        # 找到脚本文件
        scriptPath = self.task.scriptPath / (command + ".py")
        try:
            scriptModule = self.__loadAModule(scriptPath,"AScript")
            # 找到其中的experience函数
            scriptMethod = scriptModule.__getattribute__("experience")
            # experience函数固定只有self一个参数，使用self.runner填充
            scriptMethod(self.runner)
        except Exception as e:
            # 脚本出错不退出analyzer
            print(e)
            return

    def __inputLoop(self):
        self.session = PromptSession(auto_suggest=AutoSuggestFromHistory())
        while True:
            command = self.session.prompt(HTML("<skyblue><b>DLNest Analyzer>></b></skyblue>"))
            commandWordList = command.strip().split(" ")
            if commandWordList[0] == 'run':
                self.__startTest(commandWordList)
            elif commandWordList[0] == 'exit':
                break
            else:
                print("Use \'run\' to start a new analyze.")
            
    def run(self):
        sys.stdin = self.stdin
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.task.GPUID)
        if self.DEBUG:
            print("Running on GPU:",self.task.GPUID)

        self.__initAll()
        self.__inputLoop()

if __name__ == "__main__":
    ta = AnalyzeTask(
        recordPath = Path("/root/code/Phoenix_DLNest/Saves/2020-09-09_12:40:49_0"),
        scriptPath = Path("/root/code/Phoenix_DLNest/AnalyzeScripts"),
        checkpointID = 900,
        GPUID = 0
    )
    session = PromptSession(auto_suggest=AutoSuggestFromHistory())
    session.prompt("test>>")
    AP = AnalyzerProcess(ta)
    AP.stdin = sys.stdin
    tmpstdin = sys.stdin
    sys.stdin = None
    AP.start()
    AP.join()
    sys.stdin = tmpstdin
    print('here')
    session.prompt("test>>")