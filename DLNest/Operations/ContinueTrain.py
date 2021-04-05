from DLNest.Scheduler.Scheduler import Scheduler
from DLNest.Information.TrainTask import TrainTask

def continueTrain(
    recordPath : str,
    checkpointID : int = -1,
    memoryConsumption : int = -1,
    CPU : bool = False,
    DDP : bool = False,
    multiGPU : bool = False,
    description : str = "",
    otherArgs : dict = {}   
):
    devices = []
    if CPU:
        devices = [-1]
    
    task = TrainTask.fromRecord(
        recordPath = recordPath,
        checkpointID = checkpointID,
        devices = devices,
        memoryConsumption = memoryConsumption,
        multiGPU = multiGPU,
        DDP = DDP,
        description = description
    )

    scheduler = Scheduler()
    scheduler.giveATask(task,otherArgs = otherArgs)