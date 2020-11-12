from pathlib import Path
import os
import sys
import multiprocessing
import time
from multiprocessing import Process,Pipe,connection,Queue

try:
    from DLNest.Core.AnalyzeProcess import AnalyzeProcess
    from DLNest.BridgeLayers.OutputLayers import AnalyzerBuffer
    from DLNest.BridgeLayers.InformationLayer import InformationLayer,CardInfo,AnalyzeTask
except ImportError:
    sys.path.append("..")
    from .AnalyzeProcess import AnalyzeProcess
    from BridgeLayers.OutputLayers import AnalyzerBuffer
    from BridgeLayers.InformationLayer import InformationLayer,CardInfo,AnalyzeTask


class Analyzer:
    def __init__(self, timeDelay : int = 60, maxTaskPerCard = -1):
        self.analyzeProcess = None
        self.output = AnalyzerBuffer()
        self.commandPipe = None
        self.outputQueue = Queue()
        self.information = InformationLayer()
        self.timeDelay = timeDelay
        self.maxTaskPerCard = 10000 if maxTaskPerCard < 0 else maxTaskPerCard
        self.nowTask = None
    
    def __canTaskRunOnCard(self,task : AnalyzeTask, card : CardInfo):
        # 判断卡是否损坏
        if card.isBreak:
            if not card.restartCard():
                return False
        
        # 判断任务数限制是否满足
        card.checkTasks()
        if card.nowTask >= self.maxTaskPerCard:
            return False

        # 判断上次运行时间距今是否满足条件
        delta = time.time() - card.lastUseTime()
        if delta < self.timeDelay:
            return False
        
        # 若未指定显存占用，则假设使用总显存的90%
        memoryConsumption = card.totalMemory * 0.9 if task.memoryConsumption < 0 else task.memoryConsumption 
        # 判断显存是否满足限制
        freeMem = card.getFreeMemory()
        if freeMem <= memoryConsumption:
            return False
        
        # 都满足
        return True
    
    def __runTaskOnCard(self, task : AnalyzeTask, card : CardInfo):
        if not self.analyzeProcess is None:
            self.closeTask()
        
        task.GPUID = card.realID
        receiver,self.commandPipe = Pipe(False)
        self.analyzeProcess = AnalyzeProcess(task,receiver,self.outputQueue)
        self.analyzeProcess.start()
    
    def giveAnAnalyzeTask(self,task : AnalyzeTask):
        cards = self.information.usingCards()
        try:
            runningCard = None
            for card in cards:
                if self.__canTaskRunOnCard(task,card):
                    runningCard = card
                    break
            
            if runningCard is None:
                self.output.logError("No card is avaliable now.")
                return False

            self.__runTaskOnCard(task,runningCard)
            self.nowTask = task
            self.output.logMessage("Analyze Task is running now.")
            return True
        finally:
            self.information.releaseCards()
    
    def closeTask(self):
        try:
            if self.analyzeProcess is None:
                self.output.logError("No analyzer task is running. Close failed.")
                return False
            self.commandPipe.close()
            self.analyzeProcess.terminate()
            self.analyzeProcess = None
            self.commandPipe = None
            self.nowTask = None
            self.output.logMessage("Closed the analyzer task.")
            return True
        except Exception as e:
            self.output.logError(str(e))
            return False
    
    def runExp(self,command : str):
        try:
            if self.commandPipe is None:
                self.output.logIgnError("No analyzer task is running.")
                return False
            self.commandPipe.send(command)
        except Exception as e:
            self.output.logIgnError(str(e))
    
    def getOutput(self):
        while True:
            try:
                output = self.outputQueue.get(False)
                if self.output.lock.acquire():
                    try:
                        self.output.styled_text += output
                    except Exception as e:
                        print(e)
                    finally:
                        self.output.lock.release()
            except Exception as e:
                break

    def getTaskInfo(self):
        if self.nowTask is None:
            return {}
        return self.nowTask.getDict()
        