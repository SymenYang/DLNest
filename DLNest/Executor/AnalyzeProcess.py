from DLNest.Information.AnalyzeTask import AnalyzeTask
from DLNest.Executor.TaskProcess import TaskProcess
from DLNest.Output.AnalyzerBuffer import AnalyzerBuffer
from pathlib import Path
import sys
import importlib

import time
from multiprocessing import Pipe

class AnalyzeRunner:
    def __init__(self,args : dict, model , dataset, log : dict):
        self.args = args
        self.model = model
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
        if self.output != None:
            self._debugf = sys.stdout
            sys.stdout = self.output
            sys.stderr = self.output
            self.output.appName = "DLNest Analyze Process"
            self.output.isSend = True

    def runExp(self):
        self.expFunc(self.runner)

    def mainLoop(self):
        self.runner = AnalyzeRunner(self.task.args,self.model,self.dataset,self.logDict)
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

def test(self):
    print(self.args,self.model,self.dataset)
    for i in range(10):
        print(i)
        time.sleep(1)

if __name__ == "__main__":
    AB = AnalyzerBuffer()
    AT = AnalyzeTask("/root/code/DLNestTest/Saves/NOSAVE",3,devices = [-1])
    AP = AnalyzeProcess(AT,outputBuffer = AB)#,expFunc = test)
    #AP = AnalyzeProcess(AT,outputBuffer = None,expFunc = test)
    AP.start()
    lastItem = ""
    time.sleep(10)
    AT.commandQueue.put("test2")
    while True:
        now = AB.getPlainText()[1]
        if now != lastItem:
            print(now)
            lastItem = now