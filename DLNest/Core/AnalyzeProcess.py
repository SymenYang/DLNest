import os
import sys
import importlib
import shutil
import json
import torch
from pathlib import Path
import multiprocessing
from multiprocessing import Process,Pipe,connection,Queue

try:
    from DLNest.BridgeLayers.InformationLayer import AnalyzeTask
    from DLNest.BridgeLayers.OutputLayers import AnalyzerBuffer
except ImportError:
    sys.path.append("..")
    from BridgeLayers.InformationLayer import AnalyzeTask
    from BridgeLayers.OutputLayers import AnalyzerBuffer

class AnalyzeRunner:
    def __init__(self,args : dict, model , dataset):
        self.args = args
        self.model = model
        self.dataset = dataset

class AnalyzeProcess(Process):
    def __init__(self,task : AnalyzeTask,commandPipe : connection.Connection, outputQueue : Queue):
        super(AnalyzeProcess, self).__init__()
        self.task = task
        self.output = AnalyzerBuffer()
        # 如果两个path不是目录的话，直接返回
        if not (self.task.recordPath.is_dir() and self.task.scriptPath.is_dir()):
            self.output.logError("Wrong directory arguments.")
            return
        self.bufferPos = 0
        self.commandPipe = commandPipe
        self.outputQueue = outputQueue

    def __resolveAbOrRePathForAnalyze(self,filePath : Path,rootFilePath : Path):
        if filePath.is_absolute():
            return rootFilePath / (filePath.stem + filePath.suffix)
        else:
            return rootFilePath / filePath

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
        # 将保存文件夹内的文件计入sys.path
        recordDirName = str(self.task.recordPath)
        if not recordDirName in sys.path:
            sys.path.append(recordDirName)
        
        # 加载args
        with (self.task.recordPath / "args.json").open('r') as fp:
            self.args = json.load(fp)
    
        # 将model、dataset lifecycle所在的文件夹计入sys.path
        modelFilePath = self.__resolveAbOrRePathForAnalyze(Path(self.args["model_file_path"]),Path(self.args["root_file_path"]))
        datasetFilePath = self.__resolveAbOrRePathForAnalyze(Path(self.args["dataset_file_path"]),Path(self.args["root_file_path"]))
        lifeCycleFilePath = self.__resolveAbOrRePathForAnalyze(Path(self.args["life_cycle_file_path"]),Path(self.args["root_file_path"]))

        modelFileDir = str(modelFilePath.parent)
        if not modelFileDir in sys.path:
            sys.path.append(modelFileDir)
        datasetFileDir = str(datasetFilePath.parent)
        if not datasetFileDir in sys.path:
            sys.path.append(datasetFileDir)
        lifeCycleFileDir = str(lifeCycleFilePath.parent)
        if not lifeCycleFileDir in sys.path:
            sys.path.append(lifeCycleFileDir)

        # 加载model dataset lifecycle
        self.modelModule = self.__loadAModule(modelFilePath,"Model")
        self.datasetModule = self.__loadAModule(datasetFilePath,"Dataset")
        self.lifeCycleModule = self.__loadAModule(lifeCycleFilePath,"LifeCycle")
        
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
                    if ckpt_file.exists():
                        if ckpt_file.is_file():
                            stateDict = torch.load(self.args['ckpt_load'])
                            self.model.loadSaveDict(stateDict)
                            self.startEpoch = stateDict['epoch']
                    else:
                        self.output.logIgnError("Cannot load ckpt " + str(self.args['ckpt_load']) + ".")
                except Exception as e:
                    self.output.logError(str(e))
                    self.output.logError("load ckpt " + str(self.args["ckpt_load"]) + " fail, stop")
                    raise Exception("Cannot load checkpoint")
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
    
    def startTest(self,command):
        scriptPath = self.task.scriptPath / (command + ".py")
        try:
            scriptModule = self.__loadAModule(scriptPath,"AScript")
            # 找到其中的experience函数
            scriptMethod = scriptModule.__getattribute__("experience")
            # experience函数固定只有self一个参数，使用self.runner填充
            scriptMethod(self.runner)
        except Exception as e:
            # 脚本出错不退出analyzer
            self.output.logIgnError(str(e))
            return

    def __getMoreOutput(self):
        offset,styledText = self.output.getStyledText(self.bufferPos,-1)
        if styledText is None:
            self.bufferPos = offset
        else:
            self.bufferPos += len(styledText)
        return styledText

    def __inputLoop(self):
        while True:
            try:
                # 阻塞地得到命令
                command = self.commandPipe.recv()
                # 运行命令
                self.startTest(command)
                # 得到额外的输出
                styledText = self.__getMoreOutput()
                # 将输出非阻塞的放入queue中
                self.outputQueue.put(styledText,block=False)
            except Exception as e:
                print(e)
                self.output.logIgnError(str(e))

    def run(self):
        sys.stdout = self.output
        sys.stderr = self.output
        self.output.appName = "DLNest Analyze Process"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.task.GPUID)
        self.output.logMessage("Runing on GPU: " + str(self.task.GPUID))
        try:
            self.__initAll()
            self.outputQueue.put(self.__getMoreOutput(),block=False)
            self.__inputLoop()
        except Exception as e:
            self.output.logError(str(e))
            self.outputQueue.put(self.__getMoreOutput(),block=False)

if __name__ == "__main__":
    TA = AnalyzeTask(
        recordPath = "/root/code/DLNestTest/Saves/2020-11-08_21:36:54_0",
        scriptPath = "/root/code/DLNestTest/AnalyzeScripts",
        checkpointID = 208,
        GPUID=0
    )
    receiver,putter = Pipe(False)
    queue = Queue()
    AP = AnalyzeProcess(TA,receiver,queue)
    AP.start()
    putter.send("test")
    try:
        output = queue.get(True)
        print(output)
    except Exception as e:
        print(e)
    input()
    putter.send("test")
    try:
        output = queue.get(True)
        print(output)
    except Exception as e:
        print(e)
    #putter.close()
    AP.terminate()
    AP.join()