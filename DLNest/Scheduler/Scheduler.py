from DLNest.Information.TaskInformation import TaskInformation
from DLNest.Executor.AnalyzeProcess import AnalyzeProcess
from DLNest.Executor.TrainProcess import TrainProcess
from DLNest.Information.InfoCenter import InfoCenter
from DLNest.Scheduler.SchedulerStrategyBase import SchedulerStrategyBase
from DLNest.Output.AnalyzerBuffer import AnalyzerBuffer
from DLNest.Scheduler.DefaultStrategy import DefaultStrategy

from apscheduler.schedulers.background import BackgroundScheduler

class Singleton(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls, *args, **kwargs)
        return cls._instance


class Scheduler(Singleton):
    def __init__(
        self,
        strategy : SchedulerStrategyBase = None,
        timeDelay : int = 60,
        maxTaskPerDevice : int = 10000,
    ):
        self.infoCenter = InfoCenter()
        self.timeDelay = timeDelay
        self.maxTaskPerDevice = maxTaskPerDevice

        if hasattr(self,"strategy"):
            if strategy != None:
                # change strategy
                self.strategy = strategy
                self.strategy.scheduler = self
        else:
            if strategy is None:
                # set default strategy
                strategy = DefaultStrategy()
            # set strategy
            self.strategy = strategy
            self.strategy.scheduler = self
        
        if not hasattr(self,"routineScheduler"):
            self.startRoutineTask()

    def giveATask(self,task : TaskInformation, otherArgs : dict):
        print("Received task " + task.ID + ".")

        task.extraInfo = otherArgs
        self.infoCenter.addATask(task)
        self.strategy.decide(self.maxTaskPerDevice,self.timeDelay)

    def runTask(self,task : TaskInformation,devices : [int]):
        task.devices = devices
        assert task.type == "Train" or task.type == "Analyze"
        if task.type == "Train":
            TProcess = TrainProcess(task)
            task.process = TProcess
            TProcess.start()
        elif task.type == "Analyze":
            ABuffer = AnalyzerBuffer()
            task.outputBuffer = ABuffer
            AProcess = AnalyzeProcess(task,outputBuffer = ABuffer)
            task.process = AProcess
            AProcess.start()

        print(task.type + " task " + task.ID + " is runing now.")

        self.infoCenter.runATask(task)

    def __routineRun(self):
        self.strategy.decide(self.maxTaskPerDevice,self.timeDelay)

    def changeTimeDelay(self,delay):
        self.timeDelay = delay
        self.routineScheduler.remove_job(self.routineJob.id)
        self.routineJob = self.routineScheduler.add_job(self.__routineRun,"interval",seconds = self.timeDelay)

    def changeMaxTaskPerDevice(self,newValue : int):
        self.maxTaskPerDevice = newValue

    def startRoutineTask(self):
        self.routineScheduler = BackgroundScheduler()
        self.routineJob = self.routineScheduler.add_job(self.__routineRun,"interval",seconds = self.timeDelay)
        self.routineScheduler.start()