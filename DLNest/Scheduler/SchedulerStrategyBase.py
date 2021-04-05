from DLNest.Information.InfoCenter import InfoCenter
from DLNest.Information.TaskInformation import TaskInformation
from DLNest.Information.DeviceInformation import DeviceInformation

from abc import ABCMeta, abstractmethod

class SchedulerStrategyBase:
    def __init__(self):
        """
        WARNING: before a run requirement send to scheduler, infoCenter.deviceLock should be released. 
        """
        self.infoCenter = InfoCenter()
        self.scheduler = None

    @abstractmethod
    def decide(self, maxTaskPerDevice : int,maxTimeDelay : int):
        pass