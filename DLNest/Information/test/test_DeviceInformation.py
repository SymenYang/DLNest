import pytest
from DLNest.Information.DeviceInformation import DeviceInformation
from DLNest.Information.TaskInformation import TaskInformation
import time
class FakeSubprocess:
    def __init__(self,alive):
        self.alive = alive
    
    def is_alive(self):
        return self.alive

@pytest.mark.Information
class TestDeviceInformation:
    def test_init(self):
        self.DI = DeviceInformation("test")
        assert self.DI.type == "test"
        assert self.DI.nowTask == 0
        assert self.DI.totalMemory == 0
        assert self.DI.runningTask == []
        assert self.DI.isBreak == False
    
    def test_getFreeMemory(self):
        self.DI = DeviceInformation("test")
        assert self.DI.getFreeMemory() > 10000000000000000000000000
    
    def initTasks(self):
        DI = DeviceInformation("test")
        fakeAliveSubprocess = FakeSubprocess(True)
        fakeDeadSubprocess = FakeSubprocess(False)
        aliveTask = TaskInformation()
        aliveTask.process = fakeAliveSubprocess
        deadTask = TaskInformation()
        deadTask.process = fakeDeadSubprocess
        DI.addATask(aliveTask)
        DI.addATask(deadTask)
        assert DI.nowTask == 2
        assert len(DI.runningTask) == 2
        assert DI.runningTask[0] == aliveTask
        assert DI.runningTask[1] == deadTask
        return DI,aliveTask

    def test_AddATask(self):
        DI,_ = self.initTasks()

    def test_lastUseTime(self):
        t = time.time()
        DI,_ = self.initTasks()
        lt = DI.lastUseTime()
        assert lt != 0
        assert abs(lt - t) < 0.1
    
    def test_checkTasks(self):
        DI,aliveTask = self.initTasks()
        DI.checkTasks()
        assert DI.nowTask == 1
        assert len(DI.runningTask) == 1
        assert DI.runningTask[-1] == aliveTask

    def test_getTaskNum(self):
        DI,_ = self.initTasks()
        assert DI.getTaskNum() == 1

    def test_getDict(self):
        DI,aliveTask = self.initTasks()
        retDict = DI.getDict()
        assert retDict["type"] == "test"
        assert retDict["is_break"] == False
        assert retDict["num_tasks"] == 1
        for item in retDict["running_tasks"]:
            for key in item:
                assert item[key] == aliveTask.getDict()[key]
        assert isinstance(retDict["total_memory"],float) or isinstance(retDict["total_memory"],int)
        assert isinstance(retDict["free_memory"],float) or isinstance(retDict["total_memory"],int)
        t = time.time()
        assert abs(t - retDict["last_use_time"]) < 0.1