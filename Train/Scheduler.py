import time
import pynvml
import threading
import random
from pathlib import Path
from apscheduler.schedulers.background import BackgroundScheduler

try:
    from TrainProcess import TrainProcess
except ImportError:
    from .TrainProcess import TrainProcess
try:
    from Task import Task
except ImportError:
    from .Task import Task

class CardInfo:
    def __init__(self,id,realID):
        self.id = id
        self.realID = realID
        self.lastUseTime = 0.0
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
            if not item.is_alive():
                self.nowTask -= 1
            else:
                newList.append(item)
        self.runningTask = newList

    def addATask(self,newTask : TrainProcess):
        self.nowTask += 1
        self.runningTask.append(newTask)
        self.lastUseTime = time.time()

class Scheduler:
    def __init__(self,cards = [-1],timeDelay : int = 60, maxTaskPerCard = -1):
        """
        cards: 可用显卡列表，若第一个元素为-1则表示所有显卡可用
        timeDelay: 在一张显卡上两次运行至少间隔的时间差（秒），防止任务现存占用未完全的情况下安排多个任务爆显存
        maxTaskPerCard: 每张卡最多同时可跑的任务，设为-1则设置为10000，近似无限制
        """
        pynvml.nvmlInit()
        self.totalCardsInSystem = pynvml.nvmlDeviceGetCount()
        if cards[0] == -1:
            cards = [i for i in range(self.totalCardsInSystem)]
        
        self.timeDelay = timeDelay

        self.totalCards = len(cards)
        self.cards = [CardInfo(i,cards[i]) for i in range(self.totalCards)] # 虚拟卡id到真实卡id映射
        self.maxTaskPerCard =  10000 if maxTaskPerCard < 0 else maxTaskPerCard
        self.pendingTasks = []

        self.applyTaskLock = threading.Lock()

        self.nowID = 0
        self.lastTime = 0.0

    def __canTaskRunOnCard(self,task : Task, cardID : int):
        card = self.cards[cardID]
        memoryConsumption = card.totalMemory * 0.9 if task.memoryConsumption < 0 else task.memoryConsumption # 若未指定显存占用，则假设使用总显存的90%

        # 判断卡是否损坏
        if card.isBreak:
            if not card.restartCard():
                return False
        
        # 判断任务数限制是否满足
        card.checkTasks()
        if card.nowTask >= self.maxTaskPerCard:
            return False

        # 判断上次运行时间距今是否满足条件
        delta = time.time() - card.lastUseTime
        if delta < self.timeDelay:
            return False

        # 判断显存是否满足限制
        freeMem = card.getFreeMemory()
        if freeMem <= memoryConsumption:
            return False

        # 都满足
        return True

    def __runTaskOnCard(self,task : Task, cardID : int):
        
        # 获得卡
        card = self.cards[cardID]

        # 赋予时间戳
        nowTime = time.time()
        if int(nowTime) != self.lastTime: # time 是实数，需要判断整秒数相同来避免太接近的几次训练使用同一个文件夹
            self.nowID = 0
        task.timestamp = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(nowTime)) + "_" + str(self.nowID)
        self.nowID += 1
        self.lastTime = int(nowTime)
        # 赋予显卡
        task.GPUID = card.realID
        # 启动训练进程
        taskProcess = TrainProcess(task)
        taskProcess.start()
        # 向卡中添加该任务的记录
        card.addATask(taskProcess)

    def giveATask(self,task : Task, jumpInLine : bool = False):
        """
        input:
            jumpInLine: 是否插队，默认为否。如果有任务等待则加入pending list最后，没有则立即执行，若插队，则检查能否立刻进行，否则加入pending list最前
            task: 待运行的任务
        
        return:
            bool: 若为True表示已经即时运行，若为False表示未能即时运行并加入队列

        """
        if self.applyTaskLock.acquire():
            print("Scheduler received a task.")
            try:
                # 不插队，检查是否有等待
                if not jumpInLine:
                    if len(self.pendingTasks) > 0:
                        print("No card valid now, task have been appended into pending list.")
                        self.pendingTasks.append(task)
                        return False

                # 在这时，只有插队或不存在等待任务，故均直接指派
                # 检查第一个可用卡
                runningCard = None
                for id in range(len(self.cards)):
                    if self.__canTaskRunOnCard(task,id):
                        runningCard = self.cards[id]
                        break

                # 若没有可用卡，则加入等待队列，若插队，则进入等待队列首
                if runningCard is None:
                    if jumpInLine:
                        print("No card valid now, task have been put in the front of pending list.")
                        self.pendingTasks.insert(0,task)
                    else:
                        print("No card valid now, task have been appended into pending list.")
                        self.pendingTasks.append(task)
                    return False

                #有可用卡，则在可用卡运行任务
                self.__runTaskOnCard(task,runningCard.id)

                #已经开始执行 return True
                return True

            except Exception as e:
                print(e)
                return False
            finally:
                self.applyTaskLock.release()

    def __routineRunTask(self):
        """
        遍历等待队列，查看任务能否运行，可以则跑，不可以则break不查找之后的
        """
        # 与giveATask互斥
        if self.applyTaskLock.acquire():
            try:
                # 遍历
                while len(self.pendingTasks) > 0:
                    task = self.pendingTasks[0]
                    runningCardID = -1
                    for id in range(len(self.cards)):
                        if self.__canTaskRunOnCard(task,id):
                            runningCardID = id
                            break

                    # 如果当前任务不能跑，则不往后查找任务运行    
                    if runningCardID == -1:
                        break
                
                    # 运行当前任务
                    self.__runTaskOnCard(task,runningCardID)
                    # 将当前task 弹出
                    self.pendingTasks.pop(0)

            except Exception as e:
                print(e)
            finally:
                self.applyTaskLock.release()

    def startRoutineTask(self):
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(self.__routineRunTask,'interval',seconds = self.timeDelay)
        self.scheduler.start()

if __name__ == "__main__":
    sc = Scheduler(timeDelay = 10,maxTaskPerCard = 1,cards = [1,0])
    task9500 = Task(
        modelFilePath = Path("/root/code/DLNestTest/TestModel.py"),
        datasetFilePath = Path("/root/code/DLNestTest/Dataset.py"),
        lifeCycleFilePath = Path("/root/code/DLNestTest/LifeCycle.py"),
        otherFilePath = [
            Path("/root/code/DLNestTest/DatasetBase.py"),
            Path("/root/code/DLNestTest/__init__.py")
        ],
        args = {
            "save_root" : "/root/code/DLNestTest/Saves",
            "model_name" : "ModelTest",
            "dataset_name" : "DatasetTest",
            "lifeCycle_name" : "LifeCycleTest",
            "checkpoint_args" : {
                "max_ckpt_in_slow_track" : 3,
                "dilation_in_slow_track" : 2,
                "max_ckpt_in_fast_track" : 2,
                "max_ckpt_in_consistent_track" : 2
            }
        },
        description = "9500",
        memoryConsumption = 9500
    )

    task2000 = Task(
        modelFilePath = Path("/root/code/DLNestTest/TestModel.py"),
        datasetFilePath = Path("/root/code/DLNestTest/Dataset.py"),
        lifeCycleFilePath = Path("/root/code/DLNestTest/LifeCycle.py"),
        otherFilePath = [
            Path("/root/code/DLNestTest/DatasetBase.py"),
            Path("/root/code/DLNestTest/__init__.py")
        ],
        args = {
            "save_root" : "/root/code/DLNestTest/Saves",
            "model_name" : "ModelTest",
            "dataset_name" : "DatasetTest",
            "lifeCycle_name" : "LifeCycleTest",
            "checkpoint_args" : {
                "max_ckpt_in_slow_track" : 3,
                "dilation_in_slow_track" : 2,
                "max_ckpt_in_fast_track" : 2,
                "max_ckpt_in_consistent_track" : 2
            }
        },
        description = "2000",
        memoryConsumption = 2000
    )

    taskInfi = Task(
        modelFilePath = Path("/root/code/DLNestTest/TestModel.py"),
        datasetFilePath = Path("/root/code/DLNestTest/Dataset.py"),
        lifeCycleFilePath = Path("/root/code/DLNestTest/LifeCycle.py"),
        otherFilePath = [
            Path("/root/code/DLNestTest/DatasetBase.py"),
            Path("/root/code/DLNestTest/__init__.py")
        ],
        args = {
            "save_root" : "/root/code/DLNestTest/Saves",
            "model_name" : "ModelTest",
            "dataset_name" : "DatasetTest",
            "lifeCycle_name" : "LifeCycleTest",
            "checkpoint_args" : {
                "max_ckpt_in_slow_track" : 3,
                "dilation_in_slow_track" : 2,
                "max_ckpt_in_fast_track" : 2,
                "max_ckpt_in_consistent_track" : 2
            }
        },
        description = "infi"
    )

    sc.startRoutineTask()

    input()

    print(sc.giveATask(task9500))
    print(sc.pendingTasks)
    print(sc.giveATask(task2000))
    print(sc.pendingTasks)
    print(sc.giveATask(task9500))
    print(sc.pendingTasks)
    print(sc.giveATask(taskInfi))
    print(sc.pendingTasks)
    print(sc.giveATask(taskInfi))
    print(sc.pendingTasks)

    input()

    print(sc.giveATask(task9500,jumpInLine = False))
    print(sc.pendingTasks)
    print(sc.giveATask(task9500,jumpInLine = True))
    print(sc.pendingTasks)