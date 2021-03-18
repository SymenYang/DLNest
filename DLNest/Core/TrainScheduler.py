import time
import pynvml
import threading
import random
from pathlib import Path
from apscheduler.schedulers.background import BackgroundScheduler
from multiprocessing import Queue
import sys
try:
    from DLNest.BridgeLayers.OutputLayers import TrainStdout,DLNestBuffer
    from DLNest.BridgeLayers.InformationLayer import InformationLayer,CardInfo,TrainTask
    from DLNest.Core.TrainProcess import TrainProcess
    from DLNest.Core.DDPTrainProcess import DDPTrainProcess
except ImportError:
    sys.path.append("..")
    from BridgeLayers.OutputLayers import TrainStdout,DLNestBuffer
    from BridgeLayers.InformationLayer import InformationLayer,CardInfo,TrainTask
    from .TrainProcess import TrainProcess
    from .DDPTrainProcess import DDPTrainProcess

class TrainScheduler:
    def __init__(self,cards = [-1], timeDelay : int = 60, maxTaskPerCard = -1):
        """
        cards: 可用显卡列表，若第一个元素为-1则表示所有显卡可用
        timeDelay: 在一张显卡上两次运行至少间隔的时间差（秒），防止任务现存占用未完全的情况下安排多个任务爆显存
        maxTaskPerCard: 每张卡最多同时可跑的任务，设为-1则设置为10000，近似无限制
        """
        self.information = InformationLayer()
        self.information.init(cards)
        self.timeDelay = timeDelay
        self.maxTaskPerCard = 10000 if maxTaskPerCard < 0 else maxTaskPerCard

        self.nowID = 0
        self.lastTime = 0

        self.output = DLNestBuffer()
        self.DDPCount = 0

    def __canTaskRunOnCard(self,task : TrainTask, card : CardInfo):

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
    
    def __canTaskRunOnCards(self,task : TrainTask, cards : [CardInfo]):
        '''
        return [] if cannot run
        return [cards] if can run
        '''
        memoryConsumption = cards[0].totalMemory * 0.9 if task.memoryConsumption < 0 else task.memoryConsumption
        ret = []
        memInfo = []
        for item in cards:
            if item.isBreak:
                continue
        
            item.checkTasks()
            if item.nowTask >= self.maxTaskPerCard:
                continue
        
            delta = time.time() - item.lastUseTime()
            if delta < self.timeDelay:
                continue
        
            freeMem = item.getFreeMemory()
            memInfo.append((freeMem,item))
        
        memInfo.sort(key=lambda x:x[0],reverse=True)
        memCount = 0.0
        for item in memInfo:
            memCount += item[0]
            ret.append(item[1])
            if memCount > memoryConsumption:
                break

        if memCount <= memoryConsumption:
            return []
        else:
            return ret
    
    def __runTaskOnCard(self,task : TrainTask, card : CardInfo):
        # 赋予时间戳
        nowTime = time.time()
        if int(nowTime) != self.lastTime: # time 是实数，需要判断整秒数相同来避免太接近的几次训练使用同一个文件夹
            self.nowID = 0
        if task.noSave:
            task.timestamp = "NOSAVE"
        else:
            task.timestamp = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(nowTime)) + "_" + str(self.nowID)
        self.nowID += 1
        self.lastTime = int(nowTime)
        # 赋予显卡
        task.GPUID = card.realID
        # 启动训练进程
        commandQueue = Queue()
        taskProcess = TrainProcess(task,commandQueue=commandQueue)
        task.commandQueue = commandQueue
        task.status = "Running"
        taskProcess.start()
        # 向卡中添加该任务的记录
        card.addATask(taskProcess,task)
    
    def __runTaskOnCards(self, task : TrainTask, cards :[CardInfo]):
        # 获得卡ID
        cardIDs = [item.realID for item in cards]
        # 赋予时间戳
        nowTime = time.time()
        if int(nowTime) != self.lastTime: # time 是实数，需要判断整秒数相同来避免太接近的几次训练使用同一个文件夹
            self.nowID = 0
        if task.noSave:
            task.timestamp = "NOSAVE"
        else:
            task.timestamp = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(nowTime)) + "_" + str(self.nowID)
        self.nowID += 1
        self.lastTime = int(nowTime)
        # 赋予显卡
        task.GPUID = cardIDs
        # 启动训练进程
        commandQueue = Queue()
        if task.DDP:
            taskProcess = DDPTrainProcess(task,commandQueue=commandQueue, Port = str(16484 + self.DDPCount))
            self.DDPCount = (self.DDPCount + 1) % 1000
        else:
            taskProcess = TrainProcess(task,commandQueue=commandQueue)
        task.commandQueue = commandQueue
        task.status = "Running"
        taskProcess.start()
        # 向卡中添加该任务的记录
        for card in cards:
            card.addATask(taskProcess,task)
    
    def giveATask(self,task : TrainTask, jumpInLine : bool = False):
        """
        input:
            jumpInLine: 是否插队，默认为否。如果有任务等待则加入pending list最后，没有则立即执行，若插队，则检查能否立刻进行，否则加入pending list最前
            task: 待运行的任务
        
        return:
            bool: 若为True表示已经即时运行，若为False表示未能即时运行并加入队列

        """
        cards = self.information.usingCards()
        pendingTasks = self.information.usingTasks()
        try:
            if not jumpInLine:
                # 不插队，检查是否有等待
                if len(pendingTasks) > 0:
                    self.output.logMessage("No card valid now, task has been appended into pending list.",app="Train Scheduler")
                    pendingTasks.append(task)
                    return False
                
            # 在这时，只有插队或不存在等待任务，故均直接指派
            if task.multiCard:
                    # 获得分配卡集合
                    runningCards = self.__canTaskRunOnCards(task, cards)
                    
                    #没有可用卡，加入队列
                    if len(runningCards) == 0:
                        if jumpInLine:
                            self.output.logMessage("No card valid now, task has been put in the front of the pending list.",app="Train Scheduler")
                            pendingTasks.insert(0,task)
                        else:
                            self.output.logMessage("No card valid now, task has been appended into pending list.",app="Train Scheduler")
                            pendingTasks.append(task)
                        return False
                    
                    #有可用卡，则在可用卡运行任务
                    self.__runTaskOnCards(task,runningCards)
                    return True
            else:
                # 单卡情况
                # 检查第一个可用卡
                runningCard = None
                for card in cards:
                    if self.__canTaskRunOnCard(task,card):
                        runningCard = card
                        break
                
                # 若没有可用卡，则加入等待队列，若插队，则进入等待队列首
                if runningCard is None:
                    if jumpInLine:
                        self.output.logMessage("No card valid now, task has been put in the front of the pending list.",app="Train Scheduler")
                        pendingTasks.insert(0,task)
                    else:
                        self.output.logMessage("No card valid now, task has been appended into pending list.",app="Train Scheduler")
                        pendingTasks.append(task)
                    return False
                
                #有可用卡，则在可用卡运行任务
                self.__runTaskOnCard(task,runningCard)
                self.output.logMessage("Task " + task.ID + " is running on card " + str(runningCard.realID) + ".",app="Train Scheduler")
                self.output.logMessage("Task " + task.ID + " is saving to " + task.timestamp,app="Train Scheduler")
                #已经开始执行 return True
                return True
        
        except Exception as e:
            self.output.logIgnError(str(e),app="Train Scheduler")
            return False
        finally:
            self.information.releaseTasks()
            self.information.releaseCards()

    def __routineRunTask(self):
        """
        遍历等待队列，查看任务能否运行，可以则跑，不可以则break不查找之后的
        """
        cards = self.information.usingCards()
        pendingTasks = self.information.usingTasks()
        try:
            while len(pendingTasks) > 0:
                task = pendingTasks[0]
                if task.multiCard:
                    # 获得分配卡集合
                    runningCards = self.__canTaskRunOnCards(task,cards)
                    # 如果当前任务不能跑，则不往后查找任务运行
                    if len(runningCards) == 0:
                        break
                    
                    # 运行当前任务
                    self.output.logMessage("Runing task " + task.ID + " on cards",app="Train Scheduler")
                    self.__runTaskOnCards(task,runningCards)
                    # 将当前task弹出
                    pendingTasks.pop(0)
                else:
                    runningCard = None
                    for card in cards:
                        if self.__canTaskRunOnCard(task,card):
                            runningCard = card
                            break
                    
                    # 如果当前任务不能跑，则不往后查找任务运行    
                    if runningCard is None:
                        break
                    
                    # 运行当前任务
                    self.__runTaskOnCard(task,runningCard)
                    self.output.logMessage("Task " + task.ID + " is running on card " + str(runningCard.realID) + ".",app="Train Scheduler")
                    self.output.logMessage("Task " + task.ID + " is saving to " + task.timestamp,app="Train Scheduler")
                    # 将当前task弹出
                    pendingTasks.pop(0)
        except Exception as e:
            self.output.logIgnError(str(e),app="Train Scheduler")
        finally:
            self.information.releaseTasks()
            self.information.releaseCards()

    def changeTimeDelay(self,delay):
        self.timeDelay = delay
        self.scheduler.remove_job(self.schedulerJob.id)
        self.schedulerJob = self.scheduler.add_job(self.__routineRunTask,'interval',seconds = self.timeDelay)

    def startRoutineTask(self):
        self.scheduler = BackgroundScheduler()
        self.schedulerJob = self.scheduler.add_job(self.__routineRunTask,'interval',seconds = self.timeDelay)
        self.scheduler.start()

if __name__ == "__main__":
    sc = TrainScheduler(timeDelay = 10,maxTaskPerCard = 1,cards = [1,0])
    task9000_0 = TrainTask(
        args = {
        "save_root":"/root/code/DLNestTest2/Saves",
        "model_name":"Model",
        "dataset_name":"Dataset",
        "life_cycle_name":"LifeCycle",
        "checkpoint_args":{
            "max_ckpt_in_slow_track":100,
            "dilation_in_slow_track":100,
            "max_ckpt_in_fast_track":2,
            "max_ckpt_in_consistent_track":1
        },
        "root_file_path":"/root/code/DLNestTest2",
        "model_file_path":"./Model/Model.py",
        "dataset_file_path":"./Dataset/Dataset.py",
        "life_cycle_file_path":"./LifeCycle.py",
        "other_file_paths":[],
        "child_jsons":[
            "./model_config.json",
            "./dataset_config.json"
        ]
        },
        description = "9000",
        memoryConsumption=9000
    )

    task9000_1 = TrainTask(
        args = {
        "save_root":"/root/code/DLNestTest2/Saves",
        "model_name":"Model",
        "dataset_name":"Dataset",
        "life_cycle_name":"LifeCycle",
        "checkpoint_args":{
            "max_ckpt_in_slow_track":100,
            "dilation_in_slow_track":100,
            "max_ckpt_in_fast_track":2,
            "max_ckpt_in_consistent_track":1
        },
        "root_file_path":"/root/code/DLNestTest2",
        "model_file_path":"./Model/Model.py",
        "dataset_file_path":"./Dataset/Dataset.py",
        "life_cycle_file_path":"./LifeCycle.py",
        "other_file_paths":[],
        "child_jsons":[
            "./model_config.json",
            "./dataset_config.json"
        ]
        },
        description = "9000",
        memoryConsumption=9000
    )

    task9000_2 = TrainTask(
        args = {
        "save_root":"/root/code/DLNestTest2/Saves",
        "model_name":"Model",
        "dataset_name":"Dataset",
        "life_cycle_name":"LifeCycle",
        "checkpoint_args":{
            "max_ckpt_in_slow_track":100,
            "dilation_in_slow_track":100,
            "max_ckpt_in_fast_track":2,
            "max_ckpt_in_consistent_track":1
        },
        "root_file_path":"/root/code/DLNestTest2",
        "model_file_path":"./Model/Model.py",
        "dataset_file_path":"./Dataset/Dataset.py",
        "life_cycle_file_path":"./LifeCycle.py",
        "other_file_paths":[],
        "child_jsons":[
            "./model_config.json",
            "./dataset_config.json"
        ]
        },
        description = "9000",
        memoryConsumption=9000
    )

    task9000_3 = TrainTask(
        args = {
        "save_root":"/root/code/DLNestTest2/Saves",
        "model_name":"Model",
        "dataset_name":"Dataset",
        "life_cycle_name":"LifeCycle",
        "checkpoint_args":{
            "max_ckpt_in_slow_track":100,
            "dilation_in_slow_track":100,
            "max_ckpt_in_fast_track":2,
            "max_ckpt_in_consistent_track":1
        },
        "root_file_path":"/root/code/DLNestTest2",
        "model_file_path":"./Model/Model.py",
        "dataset_file_path":"./Dataset/Dataset.py",
        "life_cycle_file_path":"./LifeCycle.py",
        "other_file_paths":[],
        "child_jsons":[
            "./model_config.json",
            "./dataset_config.json"
        ]
        },
        description = "9000",
        memoryConsumption=9000
    )

    task2000 = TrainTask(
        args = {
        "save_root":"/root/code/DLNestTest2/Saves",
        "model_name":"Model",
        "dataset_name":"Dataset",
        "life_cycle_name":"LifeCycle",
        "checkpoint_args":{
            "max_ckpt_in_slow_track":100,
            "dilation_in_slow_track":100,
            "max_ckpt_in_fast_track":2,
            "max_ckpt_in_consistent_track":1
        },
        "root_file_path":"/root/code/DLNestTest2",
        "model_file_path":"./Model/Model.py",
        "dataset_file_path":"./Dataset/Dataset.py",
        "life_cycle_file_path":"./LifeCycle.py",
        "other_file_paths":[],
        "child_jsons":[
            "./model_config.json",
            "./dataset_config.json"
        ]
        },
        description = "2000",
        memoryConsumption=2000
    )

    taskInfi_0 = TrainTask(
        args = {
        "save_root":"/root/code/DLNestTest2/Saves",
        "model_name":"Model",
        "dataset_name":"Dataset",
        "life_cycle_name":"LifeCycle",
        "checkpoint_args":{
            "max_ckpt_in_slow_track":100,
            "dilation_in_slow_track":100,
            "max_ckpt_in_fast_track":2,
            "max_ckpt_in_consistent_track":1
        },
        "root_file_path":"/root/code/DLNestTest2",
        "model_file_path":"./Model/Model.py",
        "dataset_file_path":"./Dataset/Dataset.py",
        "life_cycle_file_path":"./LifeCycle.py",
        "other_file_paths":[],
        "child_jsons":[
            "./model_config.json",
            "./dataset_config.json"
        ]
        },
        description = "infi",
        memoryConsumption=-1
    )

    taskInfi_1 = TrainTask(
        args = {
        "save_root":"/root/code/DLNestTest2/Saves",
        "model_name":"Model",
        "dataset_name":"Dataset",
        "life_cycle_name":"LifeCycle",
        "checkpoint_args":{
            "max_ckpt_in_slow_track":100,
            "dilation_in_slow_track":100,
            "max_ckpt_in_fast_track":2,
            "max_ckpt_in_consistent_track":1
        },
        "root_file_path":"/root/code/DLNestTest2",
        "model_file_path":"./Model/Model.py",
        "dataset_file_path":"./Dataset/Dataset.py",
        "life_cycle_file_path":"./LifeCycle.py",
        "other_file_paths":[],
        "child_jsons":[
            "./model_config.json",
            "./dataset_config.json"
        ]
        },
        description = "infi",
        memoryConsumption=-1
    )

    sc.startRoutineTask()

    input()

    op = DLNestBuffer()

    print(sc.giveATask(task9000_0))
    print(op.getPlainText()[1])
    print(sc.information.getTasksInfo())
    print(sc.giveATask(task2000))
    print(op.getPlainText()[1])
    print(sc.information.getTasksInfo())
    print(sc.giveATask(task9000_1))
    print(op.getPlainText()[1])
    print(sc.information.getTasksInfo())
    print(sc.giveATask(taskInfi_0))
    print(op.getPlainText()[1])
    print(sc.information.getTasksInfo())
    print(sc.giveATask(taskInfi_1))
    print(op.getPlainText()[1])
    print(sc.information.getTasksInfo())

    input()

    print(sc.giveATask(task9000_2,jumpInLine = False))
    print(op.getPlainText()[1])
    print(sc.information.getTasksInfo())
    print(sc.giveATask(task9000_3,jumpInLine = True))
    print(op.getPlainText()[1])
    print(sc.information.getTasksInfo())