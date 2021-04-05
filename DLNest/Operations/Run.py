from DLNest.Scheduler.Scheduler import Scheduler
from DLNest.Information.TrainTask import TrainTask

def run(
    configPath : str,
    freqPath : str = "",
    description : str = "",
    memoryConsumption : int = -1,
    CPU : bool = False,
    DDP : bool = False,
    multiGPU : bool = False,
    noSave : bool = False,
    useDescriptionToSave : bool = False,
    otherArgs : dict = {}
):
    scheduler = Scheduler()
    
    devices = []
    if CPU:
        devices = [-1]

    task = TrainTask.fromConfigFile(
        configPath = configPath,
        freqPath = freqPath,
        description = description,
        devices = devices,
        memoryConsumption = memoryConsumption,
        multiGPU = multiGPU,
        DDP = DDP,
        noSave = noSave,
        useDescriptionToSave = useDescriptionToSave
    )

    scheduler.giveATask(task,otherArgs = otherArgs)