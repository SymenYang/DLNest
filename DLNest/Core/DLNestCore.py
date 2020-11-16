try:
    from DLNest.Core.TrainScheduler import TrainScheduler
    from DLNest.Core.Analyzer import Analyzer
    from DLNest.BridgeLayers.OutputLayers import DLNestBuffer,AnalyzerBuffer
    from DLNest.BridgeLayers.InformationLayer import InformationLayer,TrainTask,AnalyzeTask,CardInfo
except ImportError:
    import sys
    sys.path.append("..")
    from .TrainScheduler import TrainScheduler
    from .Analyzer import Analyzer
    from BridgeLayers.OutputLayers import DLNestBuffer,AnalyzerBuffer
    from BridgeLayers.InformationLayer import InformationLayer,TrainTask,AnalyzeTask,CardInfo

import shutil
import argparse
import json
import os
import signal
from pathlib import Path

class Arguments:
    def __init__(self,desc : str = ""):
        self._parser = argparse.ArgumentParser(description=desc)

    def parser(self):
        return self._parser

class DLNestArguments(Arguments):
    def __init__(self):
        super(DLNestArguments, self).__init__(desc="Arguments for DLNest trainner.")
        self._parser.add_argument("-c",type=str,default="",help="Config json for DLNest trainer")

class DLNestCore:
    def __init__(self,configFilePath = None):
        # 载入trainer的参数
        if configFilePath is None or configFilePath == "":
            configFilePath = Path(__file__).parent.parent / "DLNest_config.json"       
        trainerArgsPath = Path(configFilePath)
        fp = trainerArgsPath.open("r")
        self.trainerArgs = json.load(fp)
        fp.close()

        self.trainScheduler = TrainScheduler(
            cards = self.trainerArgs["cards"],
            timeDelay = self.trainerArgs["timeDelay"],
            maxTaskPerCard = self.trainerArgs["maxTaskPerCard"]
        )

        self.trainScheduler.startRoutineTask()

        self.analyzer = Analyzer(
            timeDelay = self.trainerArgs["timeDelay"],
            maxTaskPerCard = self.trainerArgs["maxTaskPerCard"]
        )

        self.information = InformationLayer()
        self.DLNestBuffer = DLNestBuffer()
        self.analyzerBuffer = AnalyzerBuffer()
    
    def __replaceArgs(self,newArgName,newArgValue,args):
        """
        newArgName 应该是args下的名称
        若新参数不是一个dict，或新参数不存在覆盖问题，则直接新建该参数或覆盖
        若新参数是一个dict且存在覆盖问题，则递归dict调用
        """
        if not newArgName in args:
            args[newArgName] = newArgValue
            return
        
        if not isinstance(newArgValue,dict):
            args[newArgName] = newArgValue
            return
        
        for key in newArgValue:
            self.__replaceArgs(key,newArgValue[key],args[newArgName])
        return

    def __loadArgs(self,filePath : Path,args : dict):
        
        # 若文件不存在或不是一个文件，直接返回
        if not filePath.is_file():
            return False
        try:
            fp = filePath.open('r')
            newArgs = json.load(fp)

            # 对除了child_jsons的每一个key尝试覆盖或新建args
            for key in newArgs:
                if key == "child_jsons":
                    continue
                
                # 为避免dict类型的参数被完全覆盖，使用__replaceArgs方法新建或覆盖args
                self.__replaceArgs(key,newArgs[key],args)

            # 递归查找子json，子json覆盖父json的参数（按照DFS序）
            if "child_jsons" in newArgs:
                for item in newArgs["child_jsons"]:
                    path = Path(item)
                    
                    # 若子json路径不是绝对路径，则按照当前json路径寻找相对路径
                    if not path.is_absolute():
                        path = filePath.parent / item
                    
                    # 载入子json
                    self.__loadArgs(path,args)

            fp.close()
            return True
        except Exception as e:
            self.DLNestBuffer.logIgnError(str(e))
            try:
                fp.close()
            except Exception:
                ...
            return False

    def runTrain(self,
        rootConfig : str, 
        description : str = "",
        freqConfig : str = "", 
        memoryConsumption : int = -1,
        jumpInLine : bool = False,
        multiCard : bool = False,
        noSave : bool = False
    ):
        try:
            taskArgsPath = Path(rootConfig)
            # 初始化task 参数，包括但不限于模型参数，数据集参数
            taskArgs = {}

            # 若模型参数根json获取失败，报错
            if self.__loadArgs(taskArgsPath,taskArgs):
                ...
            else:
                self.DLNestBuffer.logError("Wrong root configuration json file")
                return
        
            # 获取高频修改参数
            # 若高频修改参数获取失败，忽略
            if freqConfig != "":
                hotArgsPath = Path(freqConfig)
                self.__loadArgs(hotArgsPath,taskArgs)
            
            # 新建Task
            trainTask = TrainTask(
                args = taskArgs,
                description=description,
                memoryConsumption=memoryConsumption,
                multiCard=multiCard,
                noSave=noSave
            )

            # 运行Task
            self.DLNestBuffer.logMessage("Task " + trainTask.ID + " has been given to scheduler.")
            self.trainScheduler.giveATask(trainTask,jumpInLine)
        except Exception as e:
            self.DLNestBuffer.logError(str(e))
            self.DLNestBuffer.logError("Failed to run a task.")

    def newProj(self,targetDir : str):
        try:
            projectPath = Path(targetDir).absolute()

            # 若目标位置有文件或文件夹，失败退出
            if projectPath.exists():
                self.DLNestBuffer.logError("Path Already exists","New Project")
                return
            
            # 将FactoryFile复制进目标位置
            factoryPath = Path(__file__).parent.parent / "FactoryFiles"
            shutil.copytree(factoryPath,projectPath)
            
            # 修改root_config中的save_root与root_file_path
            rootConfigPath = projectPath / "root_config.json"
            root_config = {}
            with rootConfigPath.open('r') as fp:
                root_config = json.load(fp)
            root_config["save_root"] = str(projectPath / "Saves")
            root_config["root_file_path"] = str(projectPath)
            with rootConfigPath.open('w') as fp:
                json.dump(root_config, fp, indent=4, separators=(',', ':'))
            self.DLNestBuffer.logMessage("Created a project in " + targetDir + ".","New Project")
        except Exception as e:
            self.DLNestBuffer.logError(str(e))
            self.DLNestBuffer.logError("Fail to new a project.")
            return

    def delTask(self,taskID : str):
        try:
            taskInfo = self.information.getTasksInfo()
            delItem = None
            for item in taskInfo:
                if item["ID"] == taskID:
                    delItem = item
                    break
            
            if delItem is None:
                self.DLNestBuffer.logError("Wrong task ID to delete.",app="Del Task")
                return
            
            if delItem["status"] == "Running":
                if "pid" in delItem:
                    pid = delItem["pid"]
                    killOut = os.kill(pid, signal.SIGKILL)
                    self.DLNestBuffer.logMessage("Killed proc " + str(pid) + ", output with " + str(killOut), app="Del Task")
                else:
                    self.DLNestBuffer.logError("A Running task has no pid.",app="Del Task")
                    return
            else:
                try:
                    tasks = self.information.usingTasks()
                    delTask = None
                    for item in tasks:
                        if item.ID == taskID:
                            delTask = item
                            break
                    if delTask is None:
                        self.DLNestBuffer.logError("The task " + str(taskID) + " is not exist.",app="Del Task")
                        return
                    tasks.remove(delTask)
                    self.DLNestBuffer.logMessage("Removed the pending task " + str(taskID) + ".",app="Del Task")
                except Exception as e:
                    self.DLNestBuffer.logError(str(e),app="Del Task")
                finally:
                    self.information.releaseTasks()
        except Exception as e:
            self.DLNestBuffer.logIgnError(str(e),app="Del Task")
            return

    def loadModel(self,
        recordPath : str,
        scriptPath : str,
        checkpointID : int,
        memoryConsumption : float = -1):
        try:
            analyzeTask = AnalyzeTask(
                recordPath = recordPath,
                scriptPath = scriptPath,
                checkpointID = checkpointID,
                memoryConsumption = memoryConsumption
            )
            self.analyzer.giveAnAnalyzeTask(analyzeTask)
            self.DLNestBuffer.logMessage("An analyze task has been given to analyzer.")
        except Exception as e:
            self.DLNestBuffer.logError(str(e))
            self.DLNestBuffer.logError("Failed to run an analyze task.")
            #raise

    def releaseModel(self):
        try:
            self.analyzer.closeTask()
            self.DLNestBuffer.logMessage("Released the analyze task.")
        except Exception as e:
            self.DLNestBuffer.logError(str(e))
            self.DLNestBuffer.logError("Failed to release an analyze task.")

    def runExp(self,command):
        try:
            self.analyzer.runExp(command)
        except Exception as e:
            self.DLNestBuffer.logError(str(e))
            self.DLNestBuffer.logError("Failed to run an exp.")

    def getDLNestBuffer(self):
        return self.DLNestBuffer

    def getAnalyzerBuffer(self):
        self.analyzer.getOutput()
        return self.analyzerBuffer

    def getDLNestOutput(self):
        return self.DLNestBuffer.getPlainText()

    def getAnalyzerOutput(self):
        self.analyzer.getOutput()
        return self.analyzerBuffer.getPlainText()

    def getDLNestStyledOutput(self):
        return self.DLNestBuffer.getStyledText()

    def getAnalyzerStyledOutput(self):
        self.analyzer.getOutput()
        return self.analyzerBuffer.getStyledText()

    def getTasks(self):
        return self.information.getTasksInfo()
    
    def getAnalyzeTask(self):
        return self.analyzer.getTaskInfo()