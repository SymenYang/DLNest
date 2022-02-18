try:
    from TaskInformation import TaskInformation
    from TrainTask import TrainTask
    from AnalyzeTask import AnalyzeTask
    from DeviceInformation import DeviceInformation
    from CPUInformation import CPUInformation
except ImportError:
    from DLNest.Information.TaskInformation import TaskInformation
    from DLNest.Information.TrainTask import TrainTask
    from DLNest.Information.AnalyzeTask import AnalyzeTask
    from DLNest.Information.DeviceInformation import DeviceInformation
    from DLNest.Information.CPUInformation import CPUInformation

HAVE_GPU = True

try:
    from GPUInformation import GPUInformation
except Exception:
    try:
        from .GPUInformation import GPUInformation
    except Exception:
        HAVE_GPU = False

if HAVE_GPU:
    import pynvml

import threading

class Singleton(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls, *args, **kwargs)
        return cls._instance


class InfoCenter(Singleton):
    def __init__(self):
        if hasattr(self,'tasks'):
            return
        if HAVE_GPU:
            try:
                # Init GPUs information
                pynvml.nvmlInit()
                self.totalGPUsInSystem = pynvml.nvmlDeviceGetCount()
                self.devices = [GPUInformation(i) for i in range(self.totalGPUsInSystem)] + [CPUInformation()]
                self.availableDevices = [i for i in range(self.totalGPUsInSystem)] + [-1]
            except Exception as e:
                # using CPU only
                self.devices = [CPUInformation()]
                self.availableDevices = [0]
        else:
            # using CPU only
            self.devices = [CPUInformation()]
            self.availableDevices = [0]
        
        self.taskLock = threading.Lock()
        self.deviceLock = threading.Lock()
        self.tasks = []

    def __getAvailableDevices(self):
        """
        return the device information classes of available devices.
        """
        return [self.devices[item] for item in self.availableDevices]
    
    def usingDevicesInformation(self):
        """
        Occupy the devices information
        """
        if self.deviceLock.acquire():
            return self.__getAvailableDevices()
    
    def releaseDeviceInformation(self):
        self.deviceLock.release()
    
    def getDevicesInformation(self):
        """
        Get the informaton of all devices.
        """
        if self.deviceLock.acquire():
            try:
                return [item.getDict() for item in self.devices]
            finally:
                self.deviceLock.release()

    def getAvailableDevicesInformation(self):
        """
        Get the information of available devices in dict.
        """
        if self.deviceLock.acquire():
            try:
                availableDevices = self.__getAvailableDevices()
                return [item.getDict() for item in availableDevices]
            finally:
                self.deviceLock.release()

    def changeDevices(self,newDevicesIDList : list):
        """
        Modify available devices
        """
        newList = []
        for item in newDevicesIDList:
            if item < -1 or item >= len(self.devices):
                print("Wrong device " + str(item))
            else:
                newList.append(item)
        self.availableDevices = newList
        return True

    def __checkTasks(self):
        """
        Delete all the tasks who are supposed to be running but have no alive process.
        """
        newList = []
        for item in self.tasks:
            if item.status == "Running":
                if item.process == None or not item.process.is_alive():
                    continue
                else:
                    newList.append(item)
            else:
                newList.append(item)
        self.tasks = newList

    def getTasksInformation(self):
        """
        Get the information of all tasks in dict.
        """
        if self.taskLock.acquire():
            try:
                self.__checkTasks()
                return [item.getDict() for item in self.tasks]
            finally:
                self.taskLock.release()

    def usingTasks(self):
        """
        Occupy the tasks information.
        """
        if self.taskLock.acquire():
            return self.tasks

    def releaseTasks(self):
        """
        Release the tasks.
        """
        self.taskLock.release()

    def addATask(self,task : TaskInformation):
        """
        Add a task to the tasks list.
        """
        if self.taskLock.acquire():
            try:
                self.__checkTasks()
                self.tasks.append(task)
            finally:
                self.taskLock.release()

    def runATask(self,task : TaskInformation):
        """
        Run a task to the devices. Tasks are supposed to be in self.task already
        """
        if self.deviceLock.acquire():
            try:
                for device in task.devices:
                    task.status = "Running"
                    self.devices[device].addATask(task)
            finally:
                self.deviceLock.release()

    def getTaskByID(self,taskID : str):
        """
        Get the task information class by a taskID.
        """
        if self.taskLock.acquire():
            try:
                for item in self.tasks:
                    if item.ID == taskID:
                        return item
                return None
            finally:
                self.taskLock.release()

    def delATask(self,taskID : str):
        """
        Delete a task by taskID.
        """
        if self.taskLock.acquire():
            try:
                self.__checkTasks()
                newList = []
                for item in self.tasks:
                    if item.ID == taskID:
                        if item.process != None:
                            item.process.terminate()
                    else:
                        newList.append(item)
                self.tasks = newList
            finally:
                self.taskLock.release()

    def delAllTask(self):
        """
        Delete all task
        """
        if self.taskLock.acquire():
            try:
                self.__checkTasks()
                for task in self.tasks:
                    if task.process != None:
                        task.process.terminate()
                self.tasks = []
            finally:
                self.taskLock.release()

if __name__ == "__main__":
    IC = InfoCenter()
    print(IC.getDevicesInformation())