import time
try:
    from TaskInformation import TaskInformation
except ImportError:
    from DLNest.Information.TaskInformation import TaskInformation

class DeviceInformation:
    def __init__(self,type : str = ""):
        self.ID = -1
        self.type = type
        self.nowTask = 0
        self.totalMemory = 0
        self.runningTask = []
        self.isBreak = False
    
    def getFreeMemory(self):
        return float('inf')
    
    def checkTasks(self):
        """
        Check the tasks in this device, if one has no subprocess or the subprocess is dead, delete the task from the device
        """
        newList = []
        for item in self.runningTask:
            if item.process is None or not item.process.is_alive():
                self.nowTask -= 1
            else:
                newList.append(item)
        self.runningTask = newList
    
    def addATask(self,newTask : TaskInformation):
        """
        Add a task information to this GPU.
        """
        self.nowTask += 1
        self.runningTask.append(newTask)

    def lastUseTime(self):
        """
        Get the start time of the last task, if no running task, return 0.
        """
        self.checkTasks()
        if len(self.runningTask) > 0:
            return self.runningTask[-1].startTime
        else:
            return 0

    def getTaskNum(self):
        self.checkTasks()
        return self.nowTask

    def getDict(self):
        self.checkTasks()
        return {
            "ID" : self.ID,
            "type" : self.type,
            "is_break" : self.isBreak,
            "num_tasks" : self.nowTask,
            "running_tasks" : [item.getDict() for item in self.runningTask],
            "total_memory" : self.totalMemory,
            "free_memory" : self.getFreeMemory(),
            "last_use_time" : self.lastUseTime()
        }

    def getDeviceStr(self):
        return ""