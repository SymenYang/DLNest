from DLNest.Information.AnalyzeTask import AnalyzeTask
from DLNest.Executor.TaskProcess import TaskProcess
from DLNest.Output.AnalyzerBuffer import AnalyzerBuffer
from pathlib import Path
import sys
import importlib
import os

import time
from multiprocessing import Pipe

class AnalyzeWrapper:
    def __init__(self,args : dict, runner , dataset, log : dict):
        self.args = args
        self.runner = runner
        self.model = runner # 后向兼容
        self.dataset = dataset
        self.log = log

class AnalyzeProcess(TaskProcess):
    def __init__(self,task : AnalyzeTask, outputBuffer : AnalyzerBuffer = None, expFunc = None):
        super(AnalyzeProcess,self).__init__(task)
        self.commandQueue = task.commandQueue
        self.expFunc = expFunc
        self.output = outputBuffer

    def initOutput(self,rank = -1):
        assert rank == -1
        os.chdir(self.task.args["root_file_path"]) # Change CWD to the save package
        sys.path.append(str(self.task.scriptPath))
        if self.output != None:
            self._debugf = sys.stdout
            sys.stdout = self.output
            sys.stderr = self.output
            self.output.appName = "DLNest Analyze Process"
            self.output.isSend = True

    def runExp(self):
        self.expFunc(self.analyzeWrapper)

    def mainLoop(self):
        self.analyzeWrapper = AnalyzeWrapper(self.task.args,self.runner,self.dataset,self.logDict)
        if self.expFunc != None:
            # Have a setted exp to run
            self.runExp()
            return
        
        print("Waiting for command...")
        while True:
            try:
                command = self.commandQueue.get(block=True)
                self.startTest(command)
            except Exception as e:
                if self.output != None:
                    self.output.logIgnError(str(e))
                else:
                    print(e)

    def __loadAScript(self,filePath : Path,name : str):
        # load a script by name
        spec = importlib.util.spec_from_file_location(
            name,
            filePath
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def startTest(self,command : str):
        scriptPath = self.task.scriptPath / (command + ".py")
        try:
            scriptModule = self.__loadAScript(scriptPath,"AScript")
            # 找到其中的experience函数
            self.expFunc = scriptModule.__getattribute__("experience")
            if self.output != None:
                self.output.appName = command
            self.runExp()
        except Exception as e:
            if self.output != None:
                self.output.logIgnError(str(e))
            else:
                print(e)
            return
        finally:
            if self.output != None:
                self.output.appName = "DLNest Analyze Process"
    
    def run(self):
        try:
            super().run()
        except Exception as e:
            import traceback
            with open("./.analyzeException.tmp.txt","w") as f:
                f.write(traceback.format_exc())