import time
import pynvml
import threading
import random
from pathlib import Path
try:
    from OutputLayers import Singleton
except Exception:
    from .OutputLayers import Singleton

class CardInfo:
    def __init__(self,id,realID):
        self.id = id
        self.realID = realID
        self.nowTask = 0
        self.runningTask = []
        try:
            # 若卡信息获取失败，当作卡失效，设置isBreak = True
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(realID)
            self.meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.totalMemory = self.meminfo.total / 1024 / 1024
            self.isBreak = False
        except Exception as e:
            # 卡失效，totalMemory设为0
            self.totalMemory = 0
            self.isBreak = True

    def restartCard(self):
        """
        重新尝试得到显卡信息，若成功，则返回True，同时修改isBreak，若失败则返回False
        """
        try:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(realID)
            self.meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.totalMemory = self.meminfo.total / 1024 / 1024
            self.isBreak = False
            return True
        except Exception:
            self.isBreak = True
            return False

    def getFreeMemory(self):
        """
        return in MB
        """
        if self.isBreak:
            return 0
        try:
            self.meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return self.meminfo.free / 1024 / 1024
        except Exception as e:
            self.isBreak = True
            return 0

    def checkTasks(self):
        newList = []
        for item in self.runningTask:
            if not item[0].is_alive():
                self.nowTask -= 1
            else:
                newList.append(item)
        self.runningTask = newList

    def addATask(self,newTask,taskClass):
        self.nowTask += 1
        self.runningTask.append((newTask,time.time(),taskClass))

    def lastUseTime(self):
        self.checkTasks()
        if len(self.runningTask) > 0:
            return self.runningTask[-1][1]
        else:
            return 0

    def getDict(self):
        self.checkTasks()
        return {
            'ID' : self.id,
            'real_ID' : self.realID,
            'is_break' : self.isBreak,
            'num_tasks' : self.nowTask,
            'running_tasks' : [(item[0].pid,item[2].getDict()) for item in self.runningTask],
            'total_memory' : self.totalMemory,
            'free_memory' : self.getFreeMemory(),
            'last_use_time' : self.lastUseTime(),
        }

class TrainTask:
    def __init__(
        self,
        args : dict,
        description : str = "",
        memoryConsumption : float = -1,
        multiCard : bool = False,
        noSave : bool = False
        ):
        self.ID = "train_" + ("%.4f" % time.time())[6:] + "_" + str(random.randint(0,9))
        self.modelFilePath = Path(args["model_file_path"])
        self.datasetFilePath = Path(args["dataset_file_path"])
        self.lifeCycleFilePath = Path(args["life_cycle_file_path"])
        self.otherFilePaths = [Path(item) for item in args["other_file_paths"]]
        self.args = args
        self.description = description
        self.memoryConsumption = memoryConsumption
        self.multiCard = multiCard
        self.timestamp = ""
        self.GPUID = -1
        self.status = "Pending"
        self.noSave = noSave
        self.commandQueue = None

    def getDict(self):
        return {
            "ID" : self.ID,
            "args" : self.args,
            "description" : self.description,
            "memory_consumption" : self.memoryConsumption,
            "multi_card" : self.multiCard,
            "timestamp" : self.timestamp,
            "GPU_ID" : self.GPUID,
            "status" : self.status,
            "no_save" : self.noSave
        }

class AnalyzeTask:
    def __init__(
            self,
            recordPath : str,
            scriptPath : str,
            checkpointID : int,
            GPUID : int = 0,
            memoryConsumption : float = -1
        ):
        self.ID = "analyze_" + ("%.4f" % time.time())[6:] + "_" + str(random.randint(0,9))
        self.recordPath = Path(recordPath)
        self.scriptPath = Path(scriptPath)
        self.GPUID = GPUID
        self.checkpointID = checkpointID
        self.memoryConsumption = memoryConsumption

    def getDict(self):
        return {
            "ID" : self.ID,
            "GPU_ID" : self.GPUID,
            "checkpoint_ID" : self.checkpointID,
            "memory_consumption" : self.memoryConsumption,
            "record_path" : str(self.recordPath),
            "script_path" : str(self.scriptPath)
        }

class InformationLayer(Singleton):
    def __init__(self):
        ...

    def init(self,cards = [-1]):
        if not hasattr(self,'pendingTasks'):
            self.pendingTasks = []
    
            pynvml.nvmlInit()
            self.totalCardsInSystem = pynvml.nvmlDeviceGetCount()
            if cards[0] == -1:
                cards = [i for i in range(self.totalCardsInSystem)]
            self.totalCards = len(cards)
            self.Cards = [CardInfo(i,cards[i]) for i in range(self.totalCards)] # 虚拟卡id到真实卡id映射
            self.taskLock = threading.Lock()
            self.cardLock = threading.Lock()

    def usingCards(self):
        """
        得到显卡的信息
        """
        if self.cardLock.acquire():
            return self.Cards
    
    def changeValidCards(self,cards = []):
        if self.cardLock.acquire():
            try:
                newList = []
                already_valid = []
                for item in self.Cards:
                    if item.realID in cards:
                        newList.append(item)
                        already_valid.append(item.realID)
                    else:
                        if len(item.runningTask) != 0:
                            raise ValueError("A GPU is running task, cannot be release.")
                for item in cards:
                    if item in already_valid:
                        ...
                    else:
                        newList.append(CardInfo(len(newList),item))
                self.Cards = newList
            except Exception as e:
                raise e
            finally:
                self.cardLock.release()

    def releaseCards(self):
        self.cardLock.release()

    def getCardsInfo(self):
        """
        得到显卡的信息(以字典形式)
        """
        return [item.getDict() for item in self.Cards]

    def getTasksInfo(self):
        """
        得到所有正在运行、排队的任务的信息（以字典形式）
        """
        ret = []
        if self.cardLock.acquire():
            try:
                fromCards = []
                for card in self.Cards:
                    card.checkTasks()
                    for task in card.runningTask:
                        data = task[2].getDict()
                        data["pid"] = task[0].pid
                        fromCards.append(data)
                ret = ret + fromCards
            finally:
                self.cardLock.release()
        if self.taskLock.acquire():
            try:
                ret = ret + [item.getDict() for item in self.pendingTasks]
            finally:
                self.taskLock.release()
        return ret

    def getTaskByID(self,ID : str):
        if self.cardLock.acquire():
            try:
                for card in self.Cards:
                    card.checkTasks()
                    for task in card.runningTask:
                        id = task[2].ID # task[2] is a TrainTask class
                        if id == ID:
                            return task # (TrainProcess, start time, TrainTask)
            finally:
                self.cardLock.release()
        if self.taskLock.acquire():
            try:
                for task in self.pendingTasks:
                    if task.ID == ID:
                        return None,None,task # (TrainProcess(None), start time(None), TrainTask)
            finally:
                self.taskLock.release()
        return None # Not found

    def usingTasks(self):
        """
        占用所有正在运行、排队的任务的信息。使用之后需要调用releaseTasks
        """
        if self.taskLock.acquire():
            return self.pendingTasks

    def releaseTasks(self):
        """
        停止占用任务信息
        """
        self.taskLock.release()