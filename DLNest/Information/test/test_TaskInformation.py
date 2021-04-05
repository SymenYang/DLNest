import pytest
from DLNest.Information.TaskInformation import TaskInformation
import time

@pytest.mark.Information
class TestTaskInformation:
    def test_init(self):
        Task = TaskInformation(
            devices=[1,2,3],
            memoryConsumption = 1234,
            multiGPU = True,
            DDP = True
        )
        assert Task.DDP == True
        assert Task.devices == [1,2,3]
        assert Task.multiGPU == True
        assert Task.memoryConsumption == 1234
        assert abs(Task.startTime - time.time()) < 1.0
        assert Task.type == "None"
        assert Task.status == "Pending"
        assert Task.process == None
    
    def test_getDict(self):
        Task = TaskInformation(
            devices=[1,2,3],
            memoryConsumption = 1234,
            multiGPU = True,
            DDP = True
        )
        targetDict = {
            "devices" : [1,2,3],
            "memory_consumption" : 1234,
            "multi_GPU" : True,
            "DDP" : True,
            "type" : "None",
            "status" : "Pending"
        }
        retDict = Task.getDict()
        for key in retDict:
            assert key in targetDict or key == "ID"
            if key in targetDict:
                assert targetDict[key] == retDict[key]

