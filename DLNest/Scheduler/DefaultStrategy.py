from DLNest.Information.InfoCenter import InfoCenter
from DLNest.Scheduler.SchedulerStrategyBase import SchedulerStrategyBase
from DLNest.Information.DeviceInformation import DeviceInformation
from DLNest.Information.TaskInformation import TaskInformation

import time

class DefaultStrategy(SchedulerStrategyBase):
    def __checkTaskNumAndLastUseTime(self, device : DeviceInformation):
        """
        Check isBreak, task num, last use time
        """
        if device.isBreak:
            return False
            
        # Check task num
        taskNum = device.getTaskNum()
        if taskNum >= self.maxTaskPerDevice:
            return False
            
        # Check last use time
        delta = time.time() - device.lastUseTime()
        if delta < self.maxTimeDelay:
            return False
            
        return True

    def __canTaskRunOnDevice(self,task : TaskInformation, device : DeviceInformation):
        """
        Check free memory and last use time
        """
        if not self.__checkTaskNumAndLastUseTime(device):
            return False

        # Check free memory
        memoryConsumption = device.totalMemory * 0.9 if task.memoryConsumption < 0 else task.memoryConsumption
        freeMemory = device.getFreeMemory()
        if freeMemory <= memoryConsumption:
            return False
        
        return True

    def __canTaskRunOnDevices(self, task : TaskInformation, devices : [DeviceInformation]):
        """
        return [] if cannot
        return [ids] if can run
        won't care the CPU
        """
        memoryConsumption = devices[0].totalMemory * 0.9 if task.memoryConsumption < 0 else task.memoryConsumption
        ret = []
        memInfo = []
        for device in devices:
            # don't consider CPU and broken devices
            if device.type == "CPU" or device.isBreak:
                continue
            if not self.__checkTaskNumAndLastUseTime(device):
                continue

            freeMemory = device.getFreeMemory()
            memInfo.append((freeMemory,device))
        
        # use the devices with big free memory first
        memInfo.sort(key = lambda x:x[0], reverse = True)
        memCount = 0.0
        for item in memInfo:
            freeMemory,device = item
            ret.append(device.ID)
            memCount += freeMemory
            if memCount > memoryConsumption:
                break
        
        # return [] if no enough free memory
        if memCount <= memoryConsumption:
            return []
        else:
            return ret

    def __findARuningDevice(self, task : TaskInformation, devices : [DeviceInformation]):
        """
        return [] if cannot
        return [id] if can run
        won't care the CPU
        """
        for device in devices:
            if device.type == "CPU":
                continue
            
            if self.__canTaskRunOnDevice(task, device):
                return [device.ID]
        
        return []

    def decide(self, maxTaskPerDevice : int, maxTimeDelay : int):
        self.maxTaskPerDevice = maxTaskPerDevice
        self.maxTimeDelay = maxTimeDelay

        continueCheck = True
        while continueCheck:
            continueCheck = False
            
            tasks = self.infoCenter.usingTasks()
            try:
                numTasks = len(tasks)
                taskWait2Run = None
                for i in range(numTasks):
                    # only consider pending tasks
                    if tasks[i].status == "Pending":
                        taskWait2Run = tasks[i]
                        # only consider the oldest task
                        break
                    else:
                        continue
                    
                if taskWait2Run != None:
                    # using device informations
                    devices = self.infoCenter.usingDevicesInformation()
                    usingDevices = []
                    if taskWait2Run.devices != []:
                        # if the task has some decided devices(such as CPU), just run it.
                        usingDevices = taskWait2Run.devices
                    elif taskWait2Run.multiGPU:
                        # multi GPU logic
                        usingDevices = self.__canTaskRunOnDevices(taskWait2Run,devices)
                    else:
                        # single GPU logic
                        usingDevices = self.__findARuningDevice(taskWait2Run,devices)
                    # must release device before run task
                    self.infoCenter.releaseDeviceInformation()
                else:
                    break
                
                # if have proper device
                if usingDevices != []:
                    self.scheduler.runTask(taskWait2Run,usingDevices)
                    continueCheck = True
            finally:
                self.infoCenter.releaseTasks()
        